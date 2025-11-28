mod data;

use std::f64::consts::PI;

use autograd as ag;

use ag::ndarray as na;
use ag::ndarray::array;
use ag::tensor_ops as op;

use xsf::log_ndtr;

const HAND_BLOB_RADIUS_MIN: f64 = 1.;
const HAND_BLOB_LEN_STDDEV: f64 = 1.2;
const HAND_BLOB_CLIFF: f64 = 3.;
const KEYMASH_MISTAKE_P: f64 = 0.005;

pub fn eval_model(input: &[u8], param_vals: na::ArrayView1<f64>) -> (f64, na::Array1<f64>) {
    let input = input
        .into_iter()
        .map(|&x| {
            data::ASCII_KB_MAP
                .get(usize::from(x))
                .copied()
                .unwrap_or(-1)
        })
        .map(f64::from) // wow type safety
        .collect::<Box<[_]>>();

    let mut out_vals = ag::run::<f64, _, _>(|ctx| {
        let rmin = op::scalar(HAND_BLOB_RADIUS_MIN, ctx);
        let char_kb_map = op::convert_to_tensor(na::arr2(&data::QWERTY_CHAR_KB_MAP_DATA_ARR), ctx);
        let input = op::convert_to_tensor(na::arr1(&input), ctx);

        let params = ctx.placeholder("params", &[10]);

        let (lh_map, llen_ll) = mk_hand_blob(
            op::slice(params, &[0], &[2]),
            op::slice(params, &[2], &[4]),
            logaddexp(params.access_elem(4), rmin, ctx),
            char_kb_map,
            ctx,
        );

        let (rh_map, rlen_ll) = mk_hand_blob(
            op::slice(params, &[5], &[7]),
            op::slice(params, &[7], &[9]),
            logaddexp(params.access_elem(9), rmin, ctx),
            char_kb_map,
            ctx,
        );

        let ret = logaddexp(
            op::gather(lh_map, input, 0),
            op::gather(rh_map, input, 0),
            ctx,
        );
        let ret = ret + ((1. - KEYMASH_MISTAKE_P) / 2.).ln();
        let ret = logaddexp(ret, op::scalar(KEYMASH_MISTAKE_P.ln(), ctx), ctx);
        let ret = op::sum_all(ret);
        let ret = op::neg(ret + llen_ll + rlen_ll);

        let grad = op::grad(&[ret], &[params])[0];

        ctx.evaluator()
            .push(ret)
            .push(grad)
            .feed(params, param_vals)
            .run()
    })
    .into_iter()
    .collect::<Result<Vec<_>, _>>()
    .unwrap();

    let out_grad_vals = out_vals.pop().unwrap();
    let out_ret_vals = out_vals.pop().unwrap();

    (
        out_ret_vals.into_dimensionality::<na::Ix0>().unwrap().into_scalar(),
        out_grad_vals.into_dimensionality::<na::Ix1>().unwrap(),
    )
}

fn mk_hand_blob<'g>(
    p0_: ag::Tensor<'g, f64>,
    p1_: ag::Tensor<'g, f64>,
    r: ag::Tensor<'g, f64>,
    char_kb_map: ag::Tensor<'g, f64>,
    ctx: &'g impl ag::prelude::AsGraph<f64>,
) -> (ag::Tensor<'g, f64>, ag::Tensor<'g, f64>) {
    let p0 = p0_.reshape(&[1, 2]);
    let p1 = p1_.reshape(&[1, 2]);
    let p10 = p1 - p0;
    let l2 = op::sum_all(op::square(p10));
    let t = op::reduce_sum((char_kb_map - p0) * p10, &[-1], false) / l2;
    let t = op::clip(t, 0., 1.);
    let hmap = char_kb_map - ((op::tile(t.reshape(&[-1, 1]), 1, 2) * p10) + p0);
    let hmap = op::sqrt(op::reduce_sum(op::square(hmap), &[-1], false) + 0.01);
    let hmap = norm_logcdf((r - hmap) * HAND_BLOB_CLIFF, ctx);
    let hmap = hmap - op::reduce_logsumexp(hmap, 0, false);
    let len_ll =
        norm_logpdf((op::sqrt(l2) - 3.) / HAND_BLOB_LEN_STDDEV, ctx) - HAND_BLOB_LEN_STDDEV.ln();
    // uuuhhh so i think the gradient of op::concat is wrong
    let pos_ll0 = op::sum_all(op::neg(logaddexp(
        op::abs(p0_ - op::convert_to_tensor(array![7., 1.5], ctx))
            - op::convert_to_tensor(array![7.5, 2.], ctx),
        op::scalar(0., ctx),
        ctx,
    )));
    let pos_ll1 = op::sum_all(op::neg(logaddexp(
        op::abs(p1_ - op::convert_to_tensor(array![7., 1.5], ctx))
            - op::convert_to_tensor(array![7.5, 2.], ctx),
        op::scalar(0., ctx),
        ctx,
    )));
    return (
        op::concat(&[hmap, op::scalar(f64::NEG_INFINITY, ctx).reshape(&[1])], 0),
        len_ll + pos_ll0 + pos_ll1,
    );
}

fn norm_logcdf<'a>(
    x: ag::Tensor<'a, f64>,
    g: &'a impl ag::prelude::AsGraph<f64>,
) -> ag::Tensor<'a, f64> {
    struct NormLogCdf;
    impl ag::op::Op<f64> for NormLogCdf {
        fn compute(&self, ctx: &mut ag::op::ComputeContext<f64>) -> Result<(), ag::op::OpError> {
            let ret = ctx.input(0).mapv(log_ndtr);
            ctx.append_output(ret);
            Ok(())
        }

        fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
            let g = op::exp(norm_logpdf(ctx.input(0), ctx.graph()) - ctx.output());
            ctx.append_input_grad(Some(g * ctx.output_grad()));
        }
    }
    ag::Tensor::builder(g)
        .append_input(x, false)
        .build(NormLogCdf)
}

fn norm_logpdf<'a>(
    x: ag::Tensor<'a, f64>,
    g: &'a impl ag::prelude::AsGraph<f64>,
) -> ag::Tensor<'a, f64> {
    struct NormLogPdf;
    impl ag::op::Op<f64> for NormLogPdf {
        fn compute(&self, ctx: &mut ag::op::ComputeContext<f64>) -> Result<(), ag::op::OpError> {
            let ret = ctx.input(0).mapv(|x| -x * x / 2. - (2. * PI).sqrt().ln());
            ctx.append_output(ret);
            Ok(())
        }

        fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
            ctx.append_input_grad(Some(op::neg(ctx.input(0)) * ctx.output_grad()));
        }
    }
    ag::Tensor::builder(g)
        .append_input(x, false)
        .build(NormLogPdf)
}

fn logaddexp<'a>(
    x: ag::Tensor<'a, f64>,
    y: ag::Tensor<'a, f64>,
    g: &'a impl ag::prelude::AsGraph<f64>,
) -> ag::Tensor<'a, f64> {
    struct LogAddExp;
    impl ag::op::Op<f64> for LogAddExp {
        fn compute(&self, ctx: &mut ag::op::ComputeContext<f64>) -> Result<(), ag::op::OpError> {
            let x = ctx.input(0);
            let y = ctx.input(1);

            let ret = ag::ndarray::Zip::from(x)
                .and_broadcast(y)
                .apply_collect(|&x, &y| xsf::logaddexp(x, y));
            ctx.append_output(ret);
            Ok(())
        }

        fn grad(&self, ctx: &mut ag::op::GradientContext<f64>) {
            let ans = ctx.output();
            ctx.append_input_grad(Some(op::exp(ctx.input(0) - ans) * ctx.output_grad()));
            ctx.append_input_grad(Some(op::exp(ctx.input(1) - ans) * ctx.output_grad()));
        }
    }
    ag::Tensor::builder(g)
        .append_input(x, false)
        .append_input(y, false)
        .build(LogAddExp)
}
