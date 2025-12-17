mod data;

use dfdx::prelude::*;
use wolfe_bfgs::*;
use xsf::log_ndtr;

const HAND_BLOB_RADIUS_MIN: f64 = 1.;
const HAND_BLOB_LEN_STDDEV: f64 = 1.2;
const HAND_BLOB_CLIFF: f64 = 3.;
const KEYMASH_MISTAKE_P: f64 = 0.005;

pub fn fit_keymash_model(input: &[u8]) -> f64 {
    let input = input
        .into_iter()
        .map(|&x| {
            data::ASCII_KB_MAP
                .get(usize::from(x))
                .copied()
                .unwrap_or(-1)
                .rem_euclid(data::QWERTY_CHAR_KB_MAP_DATA_ARR_LEN + 1) as usize
        })
        .collect::<Vec<_>>();

    let dev = Cpu::default();
    let char_kb_map = dev.tensor(data::QWERTY_CHAR_KB_MAP_DATA_ARR);

    let opt_res = Bfgs::new(
        ndarray::array![1.75, 1.1, 4.75, 0.9, 1., 7.75, 0.9, 10.75, 1.1, 1.],
        |params| {
            let params: Tensor<Rank1<10>, _, _> = dev.tensor(params.to_vec());
            let g = params.alloc_grads();
            let params = params.traced(g);

            let l0 = params.c().slice((0..2,)).realize::<Rank1<2>>();
            let l1 = params.c().slice((2..4,)).realize::<Rank1<2>>();
            let lr = params.c().select::<Rank0, _>(dev.tensor(4));
            let r0 = params.c().slice((5..7,)).realize::<Rank1<2>>();
            let r1 = params.c().slice((7..9,)).realize::<Rank1<2>>();
            let rr = params.c().select::<Rank0, _>(dev.tensor(9));

            let lr = logaddexp(lr, dev.tensor(HAND_BLOB_RADIUS_MIN));
            let rr = logaddexp(rr, dev.tensor(HAND_BLOB_RADIUS_MIN));

            let (lh_map, llen_ll) = mk_hand_blob(&dev, l0, l1, lr, char_kb_map.clone());
            let (rh_map, rlen_ll) = mk_hand_blob(&dev, r0, r1, rr, char_kb_map.clone());

            let len = input.len();
            let input = dev.tensor((input.clone(), (len,)));

            let ret = logaddexp(lh_map.gather(input.clone()), rh_map.gather(input));
            let ret = ret + ((1. - KEYMASH_MISTAKE_P) / 2.).ln();
            let tmp = dev.tensor(KEYMASH_MISTAKE_P.ln()).broadcast_like(&ret);
            let ret = logaddexp(ret, tmp);
            let cost = -(ret.sum() + llen_ll + rlen_ll);

            let cost_val = cost.array();
            let out_grads = cost.backward().get(&params);
            (cost_val, ndarray::Array1::from_vec(out_grads.as_vec()))
        },
    )
    .run();

    if let Err(e) = &opt_res {
        log::warn!("Optimizer terminated with error: {}", e);
    }

    extract_soln(opt_res).map_or(f64::NEG_INFINITY, |x| -x.final_value)
}

fn extract_soln(opt_res: Result<BfgsSolution, BfgsError>) -> Option<BfgsSolution> {
    match opt_res {
        Ok(x) => Some(x),
        Err(x) => match x {
            BfgsError::LineSearchFailed { last_solution, .. } => Some(*last_solution),
            BfgsError::MaxIterationsReached { last_solution } => Some(*last_solution),
            BfgsError::GradientIsNaN => None,
            BfgsError::StepSizeTooSmall => None,
        },
    }
}

trait WithEmptyTapeShort: WithEmptyTape + Sized {
    /// Clones self and inserts a new empty tape into the clone
    fn c(&self) -> Self {
        self.with_empty_tape()
    }
}
impl<T: WithEmptyTape> WithEmptyTapeShort for T {}

fn mk_hand_blob<T: Tape<f64, Cpu> + std::fmt::Debug>(
    dev: &Cpu,
    p0: Tensor<Rank1<2>, f64, Cpu, T>,
    p1: Tensor<Rank1<2>, f64, Cpu, T>,
    r: Tensor<Rank0, f64, Cpu, T>,
    char_kb_map: Tensor<Rank2<47, 2>, f64, Cpu, NoneTape>,
) -> (Tensor<(usize,), f64, Cpu, T>, Tensor<Rank0, f64, Cpu, T>) {
    let p10 = p1.c() - p0.c();
    let l2 = p10.c().square().sum();
    let p0map = p0.c().broadcast() - char_kb_map.clone();
    let t =
        ((p0map.c() * -p10.c().broadcast()).sum::<_, Axis<1>>() / l2.c().broadcast()).clamp(0, 1);
    let hmap = ((t.broadcast::<Rank2<_, 2>, _>() * p10.broadcast() + p0map)
        .square()
        .sum::<_, Axis<1>>()
        + 0.01)
        .sqrt();
    let hmap = norm_logcdf((hmap - r.broadcast()) * -HAND_BLOB_CLIFF);
    let hmap = hmap.c() - hmap.logsumexp().broadcast();
    let len_ll = norm_logpdf((l2.sqrt() - 3.) / HAND_BLOB_LEN_STDDEV) - HAND_BLOB_LEN_STDDEV.ln();
    let pos_ll = logaddexp(
        ([p0, p1].stack() - dev.tensor([7., 1.5]).broadcast::<_, Axis<0>>()).abs()
            - dev.tensor([7.5, 2.]).broadcast::<_, Axis<0>>(),
        dev.tensor(0.).broadcast(),
    )
    .negate()
    .sum::<Rank0, _>();

    // TODO: nightly whyyyyy
    let hmap = hmap.realize::<(usize,)>();

    (
        (hmap, dev.tensor([f64::NEG_INFINITY])).concat_along(Axis),
        len_ll + pos_ll,
    )
}

fn norm_logcdf<const M: usize, T: Tape<f64, Cpu>>(
    x: Tensor<Rank1<M>, f64, Cpu, T>,
) -> Tensor<Rank1<M>, f64, Cpu, T> {
    let (x, mut tape) = x.split_tape();
    let out: Vec<_> = x.as_vec().iter().copied().map(log_ndtr).collect();
    let out = x.device().tensor(out);
    let out2 = out.clone();
    tape.add_backward_op(move |grads| {
        // TODO: this is wrong if input is a broadcasted scalar
        let out_grad = grads.get(&out2);
        let grad_x = grads.get_or_alloc_mut(&x)?;
        let grad_x_val = (norm_logpdf(x) - out2).exp() * out_grad;
        for (o, g) in grad_x.iter_mut().zip(grad_x_val.as_vec()) {
            *o += g;
        }
        Ok(())
    });
    out.put_tape(tape)
}

fn norm_logpdf<S: Shape, T: Tape<f64, Cpu>>(x: Tensor<S, f64, Cpu, T>) -> Tensor<S, f64, Cpu, T> {
    use std::f64::consts::PI;
    -x.square() / 2. - (2. * PI).sqrt().ln()
}

fn logaddexp<S: Shape, T1: Tape<f64, Cpu> + Merge<T2>, T2: Tape<f64, Cpu>>(
    a: Tensor<S, f64, Cpu, T1>,
    b: Tensor<S, f64, Cpu, T2>,
) -> Tensor<S, f64, Cpu, T1> {
    let (a, t1) = a.split_tape();
    let (b, t2) = b.split_tape();
    let mut tape = t1.merge(t2);
    let y: Vec<_> = a
        .as_vec()
        .into_iter()
        .zip(b.as_vec().into_iter())
        .map(|(a, b)| xsf::logaddexp(a, b))
        .collect();
    let out = a.device().tensor((y, a.shape().clone()));
    let out2 = out.clone();
    tape.add_backward_op(move |grads| {
        // TODO: this is wrong if either input is not contiguous
        let grad_out = grads.get(&out2);
        let grad_a = grads.get_or_alloc_mut(&a)?;
        let grad_a_val = (a - out2.clone()).exp() * grad_out.clone();
        for (o, g) in grad_a.iter_mut().zip(grad_a_val.as_vec()) {
            *o += g;
        }
        let grad_b = grads.get_or_alloc_mut(&b)?;
        let grad_b_val = (b - out2).exp() * grad_out;
        if grad_b.len() == 1 {
            // hack for when b is a broadcasted scalar
            grad_b[0] += grad_b_val.sum::<Rank0, _>().array();
        } else {
            for (o, g) in grad_b.iter_mut().zip(grad_b_val.as_vec()) {
                *o += g;
            }
        }
        Ok(())
    });
    out.put_tape(tape)
}

pub fn eval_english_model(input: &[u8]) -> f64 {
    const PRB_SPACE: f64 = 0.05;
    const PRB_MISTAKE: f64 = 0.01;

    let mut ret = 0.;
    let mut prev: &[f64; 26] = &data::ENGLISH_LETTER_FREQ;
    for &c in input {
        let (v, prev_n) = if b'a' <= c && c <= b'z' {
            let c_i = (c - b'a') as usize;
            (prev[c_i] + (-PRB_SPACE).ln_1p(), &data::ENGLISH_BIGRAM[c_i])
        } else {
            let v = if c == b' ' {
                PRB_SPACE.ln()
            } else {
                f64::NEG_INFINITY
            };
            (v, &data::ENGLISH_LETTER_FREQ)
        };
        ret += xsf::logaddexp(v + (-PRB_MISTAKE).ln_1p(), PRB_MISTAKE.ln());
        prev = prev_n;
    }
    return ret;
}

pub fn preprocess(input: &[u8]) -> Vec<u8> {
    let mut ret = Vec::with_capacity(input.len());
    if input.len() < 3 {
        ret.extend_from_slice(input);
    } else {
        ret.extend_from_slice(&input[..2]);
        for &c in &input[2..] {
            if !(ret[ret.len() - 2] == ret[ret.len() - 1] && ret[ret.len() - 1] == c) {
                ret.push(c);
            }
        }
    }
    ret.make_ascii_lowercase();
    return ret;
}
