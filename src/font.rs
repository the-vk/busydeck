#![allow(non_snake_case)]

use std::collections::HashMap;

macro_rules! collection {
    // map-like
    ($($k:expr => $v:expr),* $(,)?) => {{
        use std::iter::{Iterator, IntoIterator};
        Iterator::collect(IntoIterator::into_iter([$(($k, $v),)*]))
    }};
    // set-like
    ($($v:expr),* $(,)?) => {{
        use std::iter::{Iterator, IntoIterator};
        Iterator::collect(IntoIterator::into_iter([$($v,)*]))
    }};
}

pub fn build_font_2() -> HashMap<char, Vec<Vec<u8>>> {
    collection! {
        ' ' => space_2(),
        '0' => digit_0_2(),
        '1' => digit_1_2(),
        '2' => digit_2_2(),
        '3' => digit_3_2(),
        '4' => digit_4_2(),
        '5' => digit_5_2(),
        '6' => digit_6_2(),
        '7' => digit_7_2(),
        '8' => digit_8_2(),
        '9' => digit_9_2(),
        'C' => C_2(),
        'E' => E_2(),
        'M' => M_2(),
        'P' => P_2(),
        'U' => U_2(),
    }
}

fn space_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,0,0,0,0],
        vec![0,0,0,0,0],
        vec![0,0,0,0,0],
        vec![0,0,0,0,0],
        vec![0,0,0,0,0],
    ]
}

fn digit_0_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,1,1,0,0],
        vec![1,0,0,1,0],
        vec![1,0,0,1,0],
        vec![1,0,0,1,0],
        vec![0,1,1,0,0],
    ]
}

fn digit_1_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,0,1,1,0],
        vec![0,1,0,1,0],
        vec![0,0,0,1,0],
        vec![0,0,0,1,0],
        vec![0,0,0,1,0],
    ]
}

fn digit_2_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,1,1,1,0],
        vec![1,0,0,1,0],
        vec![0,0,1,0,0],
        vec![0,1,0,0,0],
        vec![1,1,1,1,0],
    ]
}

fn digit_3_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,1,1,0,0],
        vec![0,0,0,1,0],
        vec![0,1,1,1,0],
        vec![0,0,0,1,0],
        vec![0,1,1,0,0],
    ]
}

fn digit_4_2() -> Vec<Vec<u8>> {
    vec![
        vec![1,0,0,1,0],
        vec![1,0,0,1,0],
        vec![0,1,1,1,0],
        vec![0,0,0,1,0],
        vec![0,0,0,1,0],
    ]
}

fn digit_5_2() -> Vec<Vec<u8>> {
    vec![
        vec![1,1,1,1,0],
        vec![1,0,0,0,0],
        vec![1,1,1,1,0],
        vec![0,0,0,1,0],
        vec![1,1,1,0,0],
    ]
}

fn digit_6_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,1,1,1,0],
        vec![1,0,0,0,0],
        vec![1,1,1,0,0],
        vec![1,0,0,1,0],
        vec![1,1,1,0,0],
    ]
}

fn digit_7_2() -> Vec<Vec<u8>> {
    vec![
        vec![1,1,1,1,0],
        vec![0,0,0,1,0],
        vec![0,0,1,0,0],
        vec![0,1,0,0,0],
        vec![0,1,0,0,0],
    ]
}

fn digit_8_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,1,1,0,0],
        vec![1,0,0,1,0],
        vec![0,1,1,0,0],
        vec![1,0,0,1,0],
        vec![0,1,1,0,0],
    ]
}

fn digit_9_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,1,1,0,0],
        vec![1,0,0,1,0],
        vec![0,1,1,1,0],
        vec![0,0,0,1,0],
        vec![0,1,1,0,0],
    ]
}

fn C_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,1,1,1,0],
        vec![1,0,0,0,0],
        vec![1,0,0,0,0],
        vec![1,0,0,0,0],
        vec![0,1,1,1,0],
    ]
}

fn E_2() -> Vec<Vec<u8>> {
    vec![
        vec![0,1,1,1,0],
        vec![1,0,0,0,0],
        vec![1,1,1,0,0],
        vec![1,0,0,0,0],
        vec![0,1,1,1,0],
    ]
}

fn M_2() -> Vec<Vec<u8>> {
    vec![
        vec![1,0,0,0,1],
        vec![1,1,0,1,1],
        vec![1,0,1,0,1],
        vec![1,0,0,0,1],
        vec![1,0,0,0,1],
    ]
}

fn P_2() -> Vec<Vec<u8>> {
    vec![
        vec![1,1,1,0,0],
        vec![1,0,0,1,0],
        vec![1,0,0,1,0],
        vec![1,1,1,0,0],
        vec![1,0,0,0,0],
    ]
}

fn U_2() -> Vec<Vec<u8>> {
    vec![
        vec![1,0,0,1,0],
        vec![1,0,0,1,0],
        vec![1,0,0,1,0],
        vec![1,0,0,1,0],
        vec![0,1,1,0,0],
    ]
}