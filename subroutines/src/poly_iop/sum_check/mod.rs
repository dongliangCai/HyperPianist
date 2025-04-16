// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements the sum check protocol.

use crate::{barycentric_weights, extrapolate, poly_iop::{
    errors::PolyIOPErrors,
    structs::{IOPProof, IOPProverMessage, IOPProverState, IOPVerifierState},
    PolyIOP,
}};
use arithmetic::{build_eq_x_r, build_eq_x_r_vec, fix_variables, interpolate_uni_poly, math::Math, VPAuxInfo, VirtualPolynomial};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, end_timer, start_timer};
use ark_std::log2;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::{collections::HashMap, fmt::Debug, marker::PhantomData, sync::Arc};
use transcript::IOPTranscript;
use arithmetic::eq_poly::EqPolynomial;
use crate::index_to_field_bitvector;
use rayon::iter::IntoParallelIterator;

use deNetwork::{DeMultiNet as Net, DeNet, DeSerNet};

pub mod batched_cubic_sumcheck;
pub mod generic_sumcheck;
mod prover;
mod verifier;

/// Trait for doing sum check protocols.
pub trait SumCheck<F: PrimeField> {
    type VirtualPolynomial;
    type VPAuxInfo;
    type MultilinearExtension;

    type SumCheckProof: Clone + Debug + Default + PartialEq + CanonicalSerialize + CanonicalDeserialize;
    type Transcript;
    type SumCheckSubClaim: Clone + Debug + Default + PartialEq;

    /// Extract sum from the proof
    fn extract_sum(proof: &Self::SumCheckProof) -> F;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a SumCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// SumCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    fn prove(
        poly: Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckProof, PolyIOPErrors>;

    /// Generate proof of the sum of polynomial over {0,1}^`num_vars`
    ///
    /// The polynomial is represented in the form of a VirtualPolynomial.
    fn d_prove(
        poly: Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Option<Self::SumCheckProof>, PolyIOPErrors>;

    /// Verify the claimed sum using the proof
    fn verify(
        sum: F,
        proof: &Self::SumCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckSubClaim, PolyIOPErrors>;

    fn sum_fold(
        polys: Vec<VirtualPolynomial<F>>,
        sums: Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<(Self::SumCheckProof, F, VPAuxInfo<F>, VirtualPolynomial<F>, F), PolyIOPErrors>;
}

/// Trait for sum check protocol prover side APIs.
pub trait SumCheckProver<F: PrimeField>
where
    Self: Sized,
{
    type VirtualPolynomial;
    type ProverMessage;

    /// Initialize the prover state to argue for the sum of the input polynomial
    /// over {0,1}^`num_vars`.
    fn prover_init(polynomial: Self::VirtualPolynomial) -> Result<Self, PolyIOPErrors>;

    /// Receive message from verifier, generate prover message, and proceed to
    /// next round.
    ///
    /// Main algorithm used is from section 3.2 of [XZZPS19](https://eprint.iacr.org/2019/317.pdf#subsection.3.2).
    fn prove_round_and_update_state(
        &mut self,
        challenge: &Option<F>,
    ) -> Result<Self::ProverMessage, PolyIOPErrors>;

    /// Bind the final r_n to get the final evaluation.
    /// This should be called immediately after prove_round_and_update_state, as
    /// it assumes that the prover state has unique ownership of its polynomials
    fn get_final_mle_evaluations(&mut self, challenge: F) -> Result<Vec<F>, PolyIOPErrors>;
}

/// Trait for sum check protocol verifier side APIs.
pub trait SumCheckVerifier<F: PrimeField> {
    type VPAuxInfo;
    type ProverMessage;
    type Challenge;
    type Transcript;
    type SumCheckSubClaim;

    /// Initialize the verifier's state.
    fn verifier_init(index_info: &Self::VPAuxInfo) -> Self;

    /// Run verifier for the current round, given a prover message.
    ///
    /// Note that `verify_round_and_update_state` only samples and stores
    /// challenges; and update the verifier's state accordingly. The actual
    /// verifications are deferred (in batch) to `check_and_generate_subclaim`
    /// at the last step.
    fn verify_round_and_update_state(
        &mut self,
        prover_msg: &Self::ProverMessage,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::Challenge, PolyIOPErrors>;

    /// This function verifies the deferred checks in the interactive version of
    /// the protocol; and generate the subclaim. Returns an error if the
    /// proof failed to verify.
    ///
    /// If the asserted sum is correct, then the multilinear polynomial
    /// evaluated at `subclaim.point` will be `subclaim.expected_evaluation`.
    /// Otherwise, it is highly unlikely that those two will be equal.
    /// Larger field size guarantees smaller soundness error.
    fn check_and_generate_subclaim(
        &self,
        asserted_sum: &F,
    ) -> Result<Self::SumCheckSubClaim, PolyIOPErrors>;
}

/// A SumCheckSubClaim is a claim generated by the verifier at the end of
/// verification when it is convinced.
#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct SumCheckSubClaim<F: PrimeField> {
    /// the multi-dimensional point that this multilinear extension is evaluated
    /// to
    pub point: Vec<F>,
    /// the expected evaluation
    pub expected_evaluation: F,
}

impl<F: PrimeField> SumCheck<F> for PolyIOP<F> {
    type SumCheckProof = IOPProof<F>;
    type VirtualPolynomial = VirtualPolynomial<F>;
    type VPAuxInfo = VPAuxInfo<F>;
    type MultilinearExtension = Arc<DenseMultilinearExtension<F>>;
    type SumCheckSubClaim = SumCheckSubClaim<F>;
    type Transcript = IOPTranscript<F>;

    fn extract_sum(proof: &Self::SumCheckProof) -> F {
        let start = start_timer!(|| "extract sum");
        let res = proof.proofs[0].evaluations[0] + proof.proofs[0].evaluations[1];
        end_timer!(start);
        res
    }

    fn init_transcript() -> Self::Transcript {
        let start = start_timer!(|| "init transcript");
        let res = IOPTranscript::<F>::new(b"Initializing SumCheck transcript");
        end_timer!(start);
        res
    }

    fn prove(
        poly: Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckProof, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prove");

        transcript.append_serializable_element(b"aux info", &poly.aux_info)?;

        let num_vars = poly.aux_info.num_variables;
        let mut prover_state = IOPProverState::prover_init(poly)?;
        let mut challenge = None;
        let mut prover_msgs = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            let prover_msg =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            transcript.append_serializable_element(b"prover msg", &prover_msg)?;
            prover_msgs.push(prover_msg);
            challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);
        }
        // pushing the last challenge point to the state
        if let Some(p) = challenge {
            prover_state.challenges.push(p)
        };

        end_timer!(start);
        Ok(IOPProof {
            point: prover_state.challenges,
            proofs: prover_msgs,
        })
    }

    fn d_prove(
        poly: Self::VirtualPolynomial,
        transcript: &mut Self::Transcript,
    ) -> Result<Option<Self::SumCheckProof>, PolyIOPErrors> {
        let start = start_timer!(|| "sum check prove");

        let num_party_vars = Net::n_parties().log_2() as usize;
        if Net::am_master() {
            let mut aux_info = poly.aux_info.clone();
            aux_info.num_variables += num_party_vars;
            transcript.append_serializable_element(b"aux info", &aux_info)?;
        }

        let num_vars = poly.aux_info.num_variables;
        let max_degree = poly.aux_info.max_degree;
        let mut prover_state = IOPProverState::prover_init(poly)?;
        let mut challenge = None;
        let mut prover_msgs = Vec::with_capacity(num_vars + num_party_vars);
        for _ in 0..num_vars {
            let mut prover_msg =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            let messages = Net::send_to_master(&prover_msg);
            if Net::am_master() {
                // Sum up the subprovers' messages
                prover_msg = messages.unwrap().iter().fold(
                    IOPProverMessage {
                        evaluations: vec![F::zero(); max_degree + 1],
                    },
                    |acc, x| 
                        IOPProverMessage {
                        evaluations: acc
                            .evaluations
                            .iter()
                            .zip(x.evaluations.iter())
                            .map(|(x, y)| *x + y)
                            .collect(),
                    },
                );
                transcript.append_serializable_element(b"prover msg", &prover_msg)?;
            }
            prover_msgs.push(prover_msg);
            if Net::am_master() {
                let challenge_value = transcript.get_and_append_challenge(b"Internal round")?;
                Net::recv_from_master_uniform(Some(challenge_value));
                challenge = Some(challenge_value);
            } else {
                challenge = Some(Net::recv_from_master_uniform::<F>(None));
            }
        }

        let step = start_timer!(|| "Compute final mle");

        let final_mle_evals = IOPProverState::get_final_mle_evaluations(&mut prover_state, challenge.unwrap())?;
        let final_mle_evals = Net::send_to_master(&final_mle_evals);

        if !Net::am_master() {
            end_timer!(step);
            end_timer!(start);
            return Ok(None);
        }

        let final_mle_evals = final_mle_evals.unwrap();
        let new_mles = (0..final_mle_evals[0].len())
            .map(|poly_index| Arc::new(DenseMultilinearExtension::from_evaluations_vec(num_party_vars,
                final_mle_evals.iter().map(
                    |mle_evals| mle_evals[poly_index]
                ).collect()
            )))
            .collect();

        let mut poly = prover_state.poly.clone();
        poly.aux_info.num_variables = num_party_vars;
        poly.replace_mles(new_mles);

        end_timer!(step);

        let mut old_challenges = prover_state.challenges.clone();
        let num_vars = poly.aux_info.num_variables;
        let mut prover_state = IOPProverState::prover_init(poly)?;
        challenge = None;
        for _ in 0..num_vars {
            let prover_msg =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)?;
            transcript.append_serializable_element(b"prover msg", &prover_msg)?;
            prover_msgs.push(prover_msg);
            challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);
        }
        // pushing the last challenge point to the state
        if let Some(p) = challenge {
            prover_state.challenges.push(p)
        };

        old_challenges.append(&mut prover_state.challenges);

        end_timer!(start);
        Ok(Some(IOPProof {
            point: old_challenges,
            proofs: prover_msgs,
        }))
    }

    fn verify(
        claimed_sum: F,
        proof: &Self::SumCheckProof,
        aux_info: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::SumCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "sum check verify");

        transcript.append_serializable_element(b"aux info", aux_info)?;
        let _ = transcript.get_and_append_challenge_vectors(b"sumfold rho", aux_info.num_variables)?;
        let mut verifier_state = IOPVerifierState::verifier_init(aux_info);
        for i in 0..aux_info.num_variables {
            let prover_msg = proof.proofs.get(i).expect("proof is incomplete");
            transcript.append_serializable_element(b"prover msg", prover_msg)?;
            IOPVerifierState::verify_round_and_update_state(
                &mut verifier_state,
                prover_msg,
                transcript,
            )?;
        }

        let res = IOPVerifierState::check_and_generate_subclaim(&verifier_state, &claimed_sum);

        end_timer!(start);
        res
    }

    fn sum_fold(
        polys: Vec<VirtualPolynomial<F>>,
        sums: Vec<F>,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<(Self::SumCheckProof, F, VPAuxInfo<F>, VirtualPolynomial<F>, F), PolyIOPErrors> {
        let m = polys.len();
        let t = polys[0].flattened_ml_extensions.len();
        let num_vars = polys[0].aux_info.num_variables;
        let length = log2(m) as usize;
        //println!("length of b:{:?}", length);

        let q_aux_info = VPAuxInfo::<F> {
            max_degree: polys[0].aux_info.max_degree + 1,
            num_variables: length,
            phantom: PhantomData::default(),
        };

        transcript.append_serializable_element(b"aux info", &q_aux_info)?;
        let rho: Vec<F> = transcript.get_and_append_challenge_vectors(b"sumfold rho", length)?;
        let eq_poly = EqPolynomial::new(rho.clone());
        let eq_xr_poly = build_eq_x_r(&rho)?;
    
        //run sumcheck \sum_{b\in{0,1}^v} Q(b) = T
    
        //compute the sum T
        let mut sum_t = F::zero();
        let eq_xr_vec = eq_xr_poly.to_evaluations();
        for i in 0..m {
            sum_t += eq_xr_vec[i] * sums[i];
        }
    
        //compute evaluations of f_j(b,x)
        let new_num_vars = length + num_vars;
        let mut new_mle = Vec::new();
        let mut hm = HashMap::new();

        let eval_len = 1 << num_vars;
        for j in 0..t {
            let mut f = Vec::with_capacity(m * eval_len);
            for k in 0..eval_len {
                for i in 0..m {
                    f.push(polys[i].flattened_ml_extensions[j].evaluations[k].clone());
                }
            }
            let mle = Arc::new(DenseMultilinearExtension::from_evaluations_vec(new_num_vars, f));
            let mle_ptr = Arc::as_ptr(&mle);
            new_mle.push(mle);
            hm.insert(mle_ptr, j);
        }
        
        //compose_poly h
        let mut compose_poly = VirtualPolynomial {
            aux_info: VPAuxInfo {
                max_degree: polys[0].aux_info.max_degree + 1,
                num_variables: new_num_vars,
                phantom: PhantomData::default(),
            },
            products: polys[0].products.clone(),
            flattened_ml_extensions: new_mle,
            raw_pointers_lookup_table: hm,
        };

        // sumcheck round prove
        let mut challenge = None;
        let mut prover_msgs = Vec::with_capacity(length);
        let mut challenges = Vec::with_capacity(length);
        let mut eq_fix = eq_xr_poly.as_ref().clone();

        for round in 0..length {

            let mut flattened_ml_extensions: Vec<DenseMultilinearExtension<F>> = compose_poly
                .flattened_ml_extensions
                .par_iter()
                .map(|x| x.as_ref().clone())
                .collect();

            if let Some(chal) = challenge {
                if round == 0 {
                    return Err(PolyIOPErrors::InvalidProver(
                        "first round should be prover first.".to_string(),
                    ));
                }
                challenges.push(chal);
    
                let r = challenges[round - 1];
                #[cfg(feature = "parallel")]
                flattened_ml_extensions
                    .par_iter_mut()
                    .for_each(|mle| *mle = fix_variables(mle, &[r]));
                #[cfg(not(feature = "parallel"))]
                flattened_ml_extensions
                    .iter_mut()
                    .for_each(|mle| *mle = fix_variables(mle, &[r]));
                eq_fix = fix_variables(&eq_fix, &[r]);
            } else if round > 0 {
                return Err(PolyIOPErrors::InvalidProver(
                    "verifier message is empty".to_string(),
                ));
            }

            let products_list = compose_poly.products.clone();
            let mut products_sum = vec![F::zero(); compose_poly.aux_info.max_degree + 1];
            let extrapolation_aux:Vec<(Vec<F>, Vec<F>)> = (1..compose_poly.aux_info.max_degree)
                .map(|degree| {
                    let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
                    let weights = barycentric_weights(&points);
                    (points, weights)
                })
                .collect();

            // Step 2: generate sum for the partial evaluated polynomial:
            // f(r_1, ... r_m,, x_{m+1}... x_n)

            let mut eq_sum = vec![vec![F::zero(); 1 << (length - round - 1)]; compose_poly.aux_info.max_degree + 1];
            for b in 0..1 << (length - round - 1) {
                let table = &eq_fix;
                let mut eval = table[b << 1];
                let step  = table[(b << 1) + 1] - table[b << 1];

                eq_sum[0][b] = eval;

                eq_sum[1..].iter_mut().for_each(|acc| {
                    eval += step;
                    acc[b] = eval;
                });
            };

            products_list.iter().for_each(|(coefficient, products)| {
                let mut sum = cfg_into_iter!(0..1 << (compose_poly.aux_info.num_variables - round - 1))
                    .fold(
                        || {
                            (
                                vec![(F::zero(), F::zero()); products.len()],
                                //change products.len() + 1 to products.len() + 2 for eq(X,\rho)
                                vec![vec![F::zero(); 1 << (length - round - 1)]; products.len() + 2],
                            )
                        },
                        |(mut buf, mut acc), b| {
                            buf.iter_mut()
                                .zip(products.iter())
                                .for_each(|((eval, step), f)| {
                                    let table = &flattened_ml_extensions[*f];
                                    *eval = table[b << 1];
                                    *step = table[(b << 1) + 1] - table[b << 1];
                                });
                            //modify 0 to b % (1 << (length - round - 1))
                            //println!("b:{:?}, mod:{:?}, value:{:?}", b, (1 << (length - round - 1)), b % (1 << (length - round - 1)));
                            acc[0][b % (1 << (length - round - 1))] += buf.iter().map(|(eval, _)| eval).product::<F>();
                            acc[1..].iter_mut().for_each(|acc| {
                                buf.iter_mut().for_each(|(eval, step)| *eval += step as &_);
                                acc[b % (1 << (length - round - 1))] += buf.iter().map(|(eval, _)| eval).product::<F>();
                            });
                            (buf, acc)
                        },
                    )
                    .map(|(_, partial)| {
                        let partial_sum: Vec<F> = eq_sum[..partial.len()]
                            .iter()
                            .zip(partial.iter())
                            .map(|(eq_row, partial_row)| {
                                assert_eq!(eq_row.len(), partial_row.len());
                                eq_row
                                    .iter()
                                    .zip(partial_row)
                                    .map(|(a, b)| *a * *b)
                                    .sum::<F>()
                            })
                            .collect();
                        partial_sum
                    })
                    .reduce(
                        || vec![F::zero(); products.len() + 2],
                        |mut sum, partial_sum| {
                            sum.iter_mut()
                                .zip(partial_sum.iter())
                                .for_each(|(sum, partial_sum)| *sum += partial_sum);
                            sum
                        },
                    );
                sum.iter_mut().for_each(|sum| *sum *= coefficient);

                let extraploation = cfg_into_iter!(0..compose_poly.aux_info.max_degree - products.len() - 1)
                    .map(|i| {
                        let (points, weights) = &extrapolation_aux[products.len()];
                        let at = F::from((products.len() + 2 + i) as u64);
                        extrapolate(points, weights, &sum, &at)
                    })
                    .collect::<Vec<_>>();
                products_sum
                    .iter_mut()
                    .zip(sum.iter().chain(extraploation.iter()))
                    .for_each(|(products_sum, sum)| *products_sum += sum);
            });
            
            // update prover's state to the partial evaluated polynomial
            compose_poly.flattened_ml_extensions = flattened_ml_extensions
                .par_iter()
                .map(|x| Arc::new(x.clone()))
                .collect();

            let message = IOPProverMessage {
                evaluations: products_sum,
            };
            transcript.append_serializable_element(b"prover msg", &message)?;
            prover_msgs.push(message);
            challenge = Some(transcript.get_and_append_challenge(b"Internal round")?);
            //TODO: set challenge to 2 now for debug
            // challenge = Some(F::ONE + F::ONE);
        }
    
        // pushing the last challenge point to the state
        if let Some(p) = challenge {
            challenges.push(p);
        };

        let proof = IOPProof {
            point: challenges,
            proofs: prover_msgs,
        };
        
    
        //run sumcheck on \sum Q(b) = T   get rb and c
        // let proof = Self::prove(q_poly, transcript)?;

        let final_round_proof = proof.proofs[length - 1].evaluations.clone();
        let final_challenge = proof.point[length - 1].clone();
        let c = interpolate_uni_poly::<F>(&final_round_proof, final_challenge);
        // let c = final_round_proof[0];
        let rb = proof.point.clone();

        //compute the folded instance-witness pair
        let v = c * eq_poly.evaluate(&rb).inverse().unwrap();
        let eq_rb_vec = build_eq_x_r_vec(&rb)?;
        let mut new_mle = vec![];
        let mut hm = HashMap::new();
        for j in 0..t {
            let mut vec = vec![F::zero(); 1 << num_vars];
            // \sum_i eq_eval * f_ij(x)
            for i in 0..m {
                for (eval, sum) in polys[i].flattened_ml_extensions[j].to_evaluations().clone().iter().zip(&mut vec){
                    *sum += eq_rb_vec[i] * (*eval);
                }
            }
            let mle = Arc::new(DenseMultilinearExtension::from_evaluations_vec(num_vars, vec));
            let mle_ptr = Arc::as_ptr(&mle);
            new_mle.push(mle);
            hm.insert(mle_ptr, j);
        }
        let folded_poly = VirtualPolynomial {
            aux_info: polys[0].aux_info.clone(),
            products: polys[0].products.clone(),
            flattened_ml_extensions: new_mle,
            raw_pointers_lookup_table: hm,
        };
        
        Ok((proof, sum_t, q_aux_info, folded_poly, v))
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::{Field, UniformRand};
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::test_rng;
    use std::sync::Arc;

    fn test_sumfold(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    )-> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (poly, asserted_sum) =
            VirtualPolynomial::<Fr>::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        //println!("products:{:?}", poly.products);
        //println!("mle:{:?}", poly.flattened_ml_extensions[0].evaluations);
        let mut poly1 = poly.clone();
        poly1.flattened_ml_extensions.reverse();
        let polys = vec![poly.clone(), poly1.clone(),poly.clone(), poly1.clone(),poly.clone(), poly1.clone(),poly.clone(), poly1.clone()];
        let sums = vec![asserted_sum.clone(); 8];
        let (q_proof, q_sum, q_aux_info, fold_poly, fold_sum) = <PolyIOP<Fr> as SumCheck<Fr>>::sum_fold(polys, sums, &mut transcript)?;
        
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            q_sum,
            &q_proof,
            &q_aux_info,
            &mut transcript,
        )?;
        // let c = subclaim.expected_evaluation;
        // let rb = subclaim.point;

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(fold_poly.deep_copy(), &mut transcript)?;
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            fold_sum,
            &proof,
            &fold_poly.aux_info,
            &mut transcript,
        )?;
        assert!(
            fold_poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );

        Ok(())
    }

    fn test_sumcheck(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();

        let (poly, asserted_sum) =
            VirtualPolynomial::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        let poly_info = poly.aux_info.clone();
        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(poly.deep_copy(), &mut transcript)?;
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            asserted_sum,
            &proof,
            &poly_info,
            &mut transcript,
        )?;
        assert!(
            poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }

    fn test_sumcheck_internal(
        nv: usize,
        num_multiplicands_range: (usize, usize),
        num_products: usize,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let (poly, asserted_sum) =
            VirtualPolynomial::<Fr>::rand(nv, num_multiplicands_range, num_products, &mut rng)?;
        let poly_info = poly.aux_info.clone();
        let mut prover_state = IOPProverState::prover_init(poly.deep_copy())?;
        let mut verifier_state = IOPVerifierState::verifier_init(&poly_info);
        let mut challenge = None;
        let mut transcript = IOPTranscript::new(b"a test transcript");
        transcript
            .append_message(b"testing", b"initializing transcript for testing")
            .unwrap();
        for _ in 0..poly_info.num_variables {
            let prover_message =
                IOPProverState::prove_round_and_update_state(&mut prover_state, &challenge)
                    .unwrap();

            challenge = Some(
                IOPVerifierState::verify_round_and_update_state(
                    &mut verifier_state,
                    &prover_message,
                    &mut transcript,
                )
                .unwrap(),
            );
        }
        let subclaim =
            IOPVerifierState::check_and_generate_subclaim(&verifier_state, &asserted_sum)
                .expect("fail to generate subclaim");
        assert!(
            poly.evaluate(&subclaim.point).unwrap() == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }

    #[test]
    fn test_sum_fold_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 5;
        let num_multiplicands_range = (4, 7);
        let num_products = 1;
        test_sumfold(nv, num_multiplicands_range, num_products)
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 1;
        let num_multiplicands_range = (4, 13);
        let num_products = 5;

        test_sumcheck(nv, num_multiplicands_range, num_products)?;
        test_sumcheck_internal(nv, num_multiplicands_range, num_products)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        let nv = 12;
        let num_multiplicands_range = (4, 9);
        let num_products = 5;

        test_sumcheck(nv, num_multiplicands_range, num_products)?;
        test_sumcheck_internal(nv, num_multiplicands_range, num_products)
    }
    #[test]
    fn zero_polynomial_should_error() {
        let nv = 0;
        let num_multiplicands_range = (4, 13);
        let num_products = 5;

        assert!(test_sumcheck(nv, num_multiplicands_range, num_products).is_err());
        assert!(test_sumcheck_internal(nv, num_multiplicands_range, num_products).is_err());
    }

    #[test]
    fn test_extract_sum() -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let (poly, asserted_sum) = VirtualPolynomial::<Fr>::rand(8, (3, 4), 3, &mut rng)?;

        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(poly, &mut transcript)?;
        assert_eq!(
            <PolyIOP<Fr> as SumCheck<Fr>>::extract_sum(&proof),
            asserted_sum
        );
        Ok(())
    }

    #[test]
    /// Test that the memory usage of shared-reference is linear to number of
    /// unique MLExtensions instead of total number of multiplicands.
    fn test_shared_reference() -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let ml_extensions: Vec<_> = (0..5)
            .map(|_| Arc::new(DenseMultilinearExtension::<Fr>::rand(8, &mut rng)))
            .collect();
        let mut poly = VirtualPolynomial::new(8);
        poly.add_mle_list(
            vec![
                ml_extensions[2].clone(),
                ml_extensions[3].clone(),
                ml_extensions[0].clone(),
            ],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(
            vec![
                ml_extensions[1].clone(),
                ml_extensions[4].clone(),
                ml_extensions[4].clone(),
            ],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(
            vec![
                ml_extensions[3].clone(),
                ml_extensions[2].clone(),
                ml_extensions[1].clone(),
            ],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(
            vec![ml_extensions[0].clone(), ml_extensions[0].clone()],
            Fr::rand(&mut rng),
        )?;
        poly.add_mle_list(vec![ml_extensions[4].clone()], Fr::rand(&mut rng))?;

        assert_eq!(poly.flattened_ml_extensions.len(), 5);

        // // test memory usage for prover
        // let prover = IOPProverState::<Fr>::prover_init(&poly).unwrap();
        // assert_eq!(prover.poly.flattened_ml_extensions.len(), 5);
        // drop(prover);

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let poly_info = poly.aux_info.clone();
        let proof = <PolyIOP<Fr> as SumCheck<Fr>>::prove(poly.deep_copy(), &mut transcript)?;
        let asserted_sum = <PolyIOP<Fr> as SumCheck<Fr>>::extract_sum(&proof);

        let mut transcript = <PolyIOP<Fr> as SumCheck<Fr>>::init_transcript();
        let subclaim = <PolyIOP<Fr> as SumCheck<Fr>>::verify(
            asserted_sum,
            &proof,
            &poly_info,
            &mut transcript,
        )?;
        assert!(
            poly.evaluate(&subclaim.point)? == subclaim.expected_evaluation,
            "wrong subclaim"
        );
        Ok(())
    }
}
