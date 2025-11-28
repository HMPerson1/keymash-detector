use autograd::ndarray::array;
use keymash_detector::*;

fn main() {
    dbg!(eval_model(
        b"asdfjkl;",
        array![1.75, 1., 4.75, 1., 2., 7.75, 1., 10.75, 1., 2.].view(),
    ));
    dbg!(eval_model(
        b"asdfjkl;",
        array![1.75, 1., 4.75, 1., -1., 7.75, 1., 10.75, 1., -1.].view(),
    ));
    dbg!(eval_model(
        b"asdfjkl;",
        array![1.75, 3., 4.75, 3., 1., 7.75, 3., 10.75, 3., 1.].view(),
    ));
    dbg!(eval_model(
        b"12340987",
        array![1.75, 1., 4.75, 1., 1., 7.75, 1., 10.75, 1., 1.].view(),
    ));
    dbg!(eval_model(
        b"12340987",
        array![5.4, 3.6, 1.3, 3.2, -3.7, 9.5, 1.1, 8.9, 1.7, -0.3].view(),
    ));
}
