use std::{cmp::max, collections::BTreeMap};

use float_ord::FloatOrd;
use fxhash::FxBuildHasher;
use indexmap::IndexSet;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::Solution;

type Matrix = Vec<Vec<(usize, usize)>>;
type MatrixPriority = (isize, FloatOrd<f32>);

enum MatrixCompatibility {
    Compatible(isize, isize, MatrixPriority),
    Incompatible,
    NotRelated,
}

const CANDIDATE_MATCH_RATIO: f32 = 1.15;
const CANDIDATE_MATCH_RATIO_EQUAL: f32 = 1.0001;
const MAX_CANDIDATES: usize = 10;
const MAX_CANDIDATES_EQUAL: usize = 7;

pub fn find_match_candidates(
    img_width: usize,
    pieces_compatibility: &[Vec<Vec<f32>>; 2],
) -> [Vec<Vec<(usize, usize)>>; 2] {
    let get_candidates = |ind: usize| {
        let min_scores = pieces_compatibility[ind]
            .par_iter()
            .enumerate()
            .map(|(i, v)| {
                v.iter()
                    .enumerate()
                    .filter(|(j, _)| i != *j)
                    .reduce(|x, y| if x.1 <= y.1 { x } else { y })
                    .map(|(_, x)| *x)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        pieces_compatibility[ind]
            .par_iter()
            .zip(min_scores)
            .enumerate()
            .map(|(i, (v, min_score))| {
                let mut tmp: Vec<_> = v
                    .iter()
                    .enumerate()
                    .filter(|(j, x)| i != *j && **x <= CANDIDATE_MATCH_RATIO * min_score)
                    .collect();
                tmp.sort_by_key(|x| FloatOrd(*x.1));
                tmp.into_iter()
                    .take(MAX_CANDIDATES)
                    .enumerate()
                    .filter_map(|(ind, x)| {
                        if ind < MAX_CANDIDATES_EQUAL
                            || *x.1 > CANDIDATE_MATCH_RATIO_EQUAL * min_score
                        {
                            Some(x)
                        } else {
                            None
                        }
                    })
                    .map(|(j, _)| (j / img_width, j % img_width))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let right_candidates = get_candidates(0);
    let down_candidates = get_candidates(1);

    let get_opposite_candidates = |ind: usize| {
        let min_scores = (0..pieces_compatibility[ind].len())
            .into_par_iter()
            .map(|i| {
                pieces_compatibility[ind]
                    .iter()
                    .enumerate()
                    .map(|(j, v)| (j, v[i]))
                    .filter(|(j, _)| i != *j)
                    .reduce(|x, y| if x.1 <= y.1 { x } else { y })
                    .map(|(_, x)| x)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        (0..pieces_compatibility[ind].len())
            .into_par_iter()
            .zip(min_scores)
            .map(|(i, min_score)| {
                let mut tmp: Vec<_> = pieces_compatibility[ind]
                    .iter()
                    .enumerate()
                    .map(|(j, v)| (j, v[i]))
                    .filter(|(j, x)| i != *j && *x <= CANDIDATE_MATCH_RATIO * min_score)
                    .collect();
                tmp.sort_by_key(|x| FloatOrd(x.1));
                tmp.into_iter()
                    .take(MAX_CANDIDATES)
                    .enumerate()
                    .filter_map(|(ind, x)| {
                        if ind < MAX_CANDIDATES_EQUAL
                            || x.1 > CANDIDATE_MATCH_RATIO_EQUAL * min_score
                        {
                            Some(x)
                        } else {
                            None
                        }
                    })
                    .map(|(j, _)| (j / img_width, j % img_width))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let left_candidates = get_opposite_candidates(0);
    let up_candidates = get_opposite_candidates(1);

    let get_buddies = |candidates: Vec<Vec<(usize, usize)>>,
                       opposite_candidates: Vec<Vec<(usize, usize)>>| {
        candidates
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let x = (i / img_width, i % img_width);
                v.into_iter()
                    .filter(|y| opposite_candidates[y.0 * img_width + y.1].contains(&x))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let right_buddies = get_buddies(right_candidates, left_candidates);
    let down_buddies = get_buddies(down_candidates, up_candidates);

    [right_buddies, down_buddies]
}

fn small_loops_matches(size: usize, sl: &[Matrix]) -> [Vec<Vec<usize>>; 2] {
    let right_matches = sl
        .iter()
        .map(|sl_left| {
            sl.iter()
                .enumerate()
                .filter_map(|(sl_right_i, sl_right)| {
                    for r in 0..size {
                        for c in 0..(size - 1) {
                            if sl_left[r][c + 1] != sl_right[r][c] {
                                return None;
                            }
                        }
                    }
                    for v_left in sl_left {
                        for v_right in sl_right {
                            if v_left[0] == v_right[size - 1] {
                                return None;
                            }
                        }
                    }
                    Some(sl_right_i)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let down_matches = sl
        .iter()
        .map(|sl_up| {
            sl.iter()
                .enumerate()
                .filter_map(|(sl_down_i, sl_down)| {
                    for r in 0..(size - 1) {
                        for c in 0..size {
                            if sl_up[r + 1][c] != sl_down[r][c] {
                                return None;
                            }
                        }
                    }
                    for x_left in &sl_up[0] {
                        for x_right in &sl_down[size - 1] {
                            if x_left == x_right {
                                return None;
                            }
                        }
                    }
                    Some(sl_down_i)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    [right_matches, down_matches]
}

fn merge_loops(
    size: usize,
    left_up: &Matrix,
    right_up: &Matrix,
    left_down: &Matrix,
    right_down: &Matrix,
) -> Matrix {
    let mut new_loop = vec![vec![(usize::MAX, usize::MAX); size + 1]; size + 1];
    for r in 0..size {
        for c in 0..size {
            new_loop[r][c] = left_up[r][c];
        }
    }
    for r in 0..size {
        new_loop[r][size] = right_up[r][size - 1];
    }
    for c in 0..size {
        new_loop[size][c] = left_down[size - 1][c];
    }
    new_loop[size][size] = right_down[size - 1][size - 1];
    new_loop
}

fn small_loops(
    img_width: usize,
    img_height: usize,
    pieces_match_candidates: &[Vec<Vec<(usize, usize)>>; 2],
) -> Vec<Vec<Matrix>> {
    let sl_1_right = (0..img_height).flat_map(|left_r| {
        (0..img_width)
            .flat_map(|left_c| {
                let left_i = left_r * img_width + left_c;
                pieces_match_candidates[0][left_i]
                    .iter()
                    .map(|right| vec![vec![(left_r, left_c), *right]])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    let sl_1_down = (0..img_height).flat_map(|up_r| {
        (0..img_width)
            .flat_map(|up_c| {
                let up_i = up_r * img_width + up_c;
                pieces_match_candidates[1][up_i]
                    .iter()
                    .map(|down| vec![vec![(up_r, up_c)], vec![*down]])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    let sl_1 = sl_1_right.chain(sl_1_down).collect::<Vec<_>>();

    let sl_2 = (0..img_height)
        .flat_map(|left_up_r| {
            (0..img_width)
                .flat_map(|left_up_c| {
                    let left_up_i = left_up_r * img_width + left_up_c;
                    pieces_match_candidates[0][left_up_i]
                        .iter()
                        .flat_map(|right_up| {
                            let right_up_i = right_up.0 * img_width + right_up.1;
                            if right_up_i == left_up_i {
                                return Vec::new();
                            }

                            pieces_match_candidates[1][left_up_i]
                                .iter()
                                .flat_map(|left_down| {
                                    let left_down_i = left_down.0 * img_width + left_down.1;
                                    if left_down_i == left_up_i || left_down_i == right_up_i {
                                        return Vec::new();
                                    }

                                    pieces_match_candidates[0][left_down_i]
                                        .iter()
                                        .flat_map(|right_down_0| {
                                            if *right_down_0 == (left_up_r, left_up_c)
                                                || right_down_0 == right_up
                                                || right_down_0 == left_down
                                            {
                                                return Vec::new();
                                            }
                                            pieces_match_candidates[1][right_up_i]
                                                .iter()
                                                .find_map(|right_down_1| {
                                                    if right_down_0 == right_down_1 {
                                                        Some(vec![
                                                            vec![(left_up_r, left_up_c), *right_up],
                                                            vec![*left_down, *right_down_0],
                                                        ])
                                                    } else {
                                                        None
                                                    }
                                                })
                                                .into_iter()
                                                .collect::<Vec<_>>()
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut sl_all = vec![sl_1, sl_2];
    loop {
        let sl_last = sl_all.last().unwrap();
        let sl_size = sl_all.len();
        let matches = small_loops_matches(sl_size, sl_last);
        let sl_next = (0..sl_last.len())
            .flat_map(|left_up_i| {
                matches[0][left_up_i]
                    .iter()
                    .flat_map(|right_up_i| {
                        matches[1][left_up_i]
                            .iter()
                            .flat_map(|left_down_i| {
                                matches[0][*left_down_i]
                                    .iter()
                                    .flat_map(|right_down_0_i| {
                                        matches[1][*right_up_i]
                                            .iter()
                                            .filter_map(|right_down_1_i| {
                                                if right_down_0_i == right_down_1_i
                                                    && sl_last[left_up_i][0][0]
                                                        != sl_last[*right_down_0_i][sl_size - 1]
                                                            [sl_size - 1]
                                                    && sl_last[*right_up_i][0][sl_size - 1]
                                                        != sl_last[*left_down_i][sl_size - 1][0]
                                                {
                                                    let sl = merge_loops(
                                                        sl_size,
                                                        &sl_last[left_up_i],
                                                        &sl_last[*right_up_i],
                                                        &sl_last[*left_down_i],
                                                        &sl_last[*right_down_0_i],
                                                    );
                                                    Some(sl)
                                                } else {
                                                    None
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if sl_next.is_empty() {
            break;
        }
        sl_all.push(sl_next);
    }
    sl_all
}

// Приоритет матрицы
fn matrix_priority(
    img_width: usize,
    pieces_compatibility: &[Vec<Vec<f32>>; 2],
    m: &Matrix,
) -> MatrixPriority {
    let cnt = m.iter().flatten().filter(|x| x.0 != usize::MAX).count();

    // Совместимости деталей по направлению вправо
    let right_compatibility = (0..m.len())
        .map(|i_r| {
            (0..(m[i_r].len() - 1))
                .map(|i_c| {
                    let (j_r, j_c) = m[i_r][i_c];
                    let (k_r, k_c) = m[i_r][i_c + 1];
                    if j_r == usize::MAX || k_r == usize::MAX {
                        return 0.0;
                    }
                    pieces_compatibility[0][j_r * img_width + j_c][k_r * img_width + k_c]
                })
                .sum::<f32>()
        })
        .sum::<f32>();
    // Совместимости деталей по направлению вниз
    let down_compatibility = (0..(m.len() - 1))
        .map(|i_r| {
            (0..m[i_r].len())
                .map(|i_c| {
                    let (j_r, j_c) = m[i_r][i_c];
                    let (k_r, k_c) = m[i_r + 1][i_c];
                    if j_r == usize::MAX || k_r == usize::MAX {
                        return 0.0;
                    }
                    pieces_compatibility[1][j_r * img_width + j_c][k_r * img_width + k_c]
                })
                .sum::<f32>()
        })
        .sum::<f32>();
    (
        -(cnt as isize),
        FloatOrd((right_compatibility + down_compatibility) / (cnt as f32)),
    )
}

// Проверка матриц на совместимость (возможность объединения)
fn can_merge_matrices(
    img_width: usize,
    img_height: usize,
    pieces_compatibility: &[Vec<Vec<f32>>; 2],
    m_x: &Matrix,
    m_y: &Matrix,
) -> MatrixCompatibility {
    // Отсортированные детали второй матрицы
    let mut tmp_y: Vec<_> = m_y
        .iter()
        .flatten()
        .cloned()
        .enumerate()
        .filter_map(|(i, x)| {
            if x.0 != usize::MAX {
                Some((x, i))
            } else {
                None
            }
        })
        .collect();
    tmp_y.sort_unstable();
    // Количество деталей, встречающихся в обоих матрицах
    let mut cnt = 0;
    // Сдвиг между общими деталями в первой и второй матрицах
    let mut shared_shift = None;
    // Проверка существования детали из первой матрицы во второй
    for (x_ind, x) in m_x.iter().flatten().enumerate() {
        if x.0 == usize::MAX {
            continue;
        }
        if let Ok(y_ind_tmp) = tmp_y.binary_search_by_key(&x, |pr| &pr.0) {
            cnt += 1;
            // Вычисление общего сдвига
            if shared_shift.is_none() {
                let y_ind = tmp_y[y_ind_tmp].1;
                let (x_r, x_c) = (x_ind / m_x[0].len(), x_ind % m_x[0].len());
                let (y_r, y_c) = (y_ind / m_y[0].len(), y_ind % m_y[0].len());
                shared_shift = Some((
                    (y_r as isize) - (x_r as isize),
                    (y_c as isize) - (x_c as isize),
                ));
            }
            // Для проверки достаточно двух общих деталей
            if cnt == 2 {
                break;
            }
        }
    }
    // Если не больше одной общей детали, то матрицы не связаны друг с другом
    // Исключение для матриц-пар
    if cnt < 2
        && !(cnt == 1
            && ((m_x.len() == 1 || m_x[0].len() == 1) ^ (m_y.len() == 1 || m_y[0].len() == 1)))
    {
        return MatrixCompatibility::NotRelated;
    }
    let (shared_shift_r, shared_shift_c) = shared_shift.unwrap();

    // Размеры матриц x и y
    let (m_x_r, m_x_c) = (m_x.len(), m_x[0].len());
    let (m_y_r, m_y_c) = (m_y.len(), m_y[0].len());
    // Переход из системы координат матрицы x к с. к. новой матрицы
    let (d_r, d_c) = (
        max(0, shared_shift_r) as usize,
        max(0, shared_shift_c) as usize,
    );
    // Переход из системы координат матрицы y к с. к. новой матрицы
    let (d_minus_shift_r, d_minus_shift_c) = (
        ((d_r as isize) - shared_shift_r) as usize,
        ((d_c as isize) - shared_shift_c) as usize,
    );
    // Размер новой матрицы
    let (m_new_r, m_new_c) = (
        max(m_x_r + d_r, m_y_r + d_minus_shift_r),
        max(m_x_c + d_c, m_y_c + d_minus_shift_c),
    );
    // Матрица должна быть не больше изображения
    if m_new_r > img_height || m_new_c > img_width {
        return MatrixCompatibility::NotRelated;
    }

    // Новая матрица
    let mut m_new = vec![vec![(usize::MAX, usize::MAX); m_new_c]; m_new_r];
    // Копирование в неё матрицы x
    for (r_x, v_x) in m_x.iter().enumerate() {
        for (c_x, x) in v_x.iter().enumerate() {
            let (r_new, c_new) = (r_x + d_r, c_x + d_c);
            m_new[r_new][c_new] = *x;
        }
    }
    // Копирование в неё матрицы y
    for (r_y, v_y) in m_y.iter().enumerate() {
        for (c_y, y) in v_y.iter().enumerate() {
            let (r_new, c_new) = (r_y + d_minus_shift_r, c_y + d_minus_shift_c);
            if m_new[r_new][c_new].0 != usize::MAX && m_new[r_new][c_new] != *y {
                return MatrixCompatibility::Incompatible;
            }
            m_new[r_new][c_new] = *y;
        }
    }
    // Если новая матрица совпадает с одной из объединяемых, то объединять не нужно
    if m_new == *m_x || m_new == *m_y {
        return MatrixCompatibility::Incompatible;
    }

    // Приоритет новой матрицы
    let m_new_priority = matrix_priority(img_width, pieces_compatibility, &m_new);

    // Отсортированные детали новой матрицы
    let mut tmp_new: Vec<_> = m_new
        .into_iter()
        .flatten()
        .filter(|x| x.0 != usize::MAX)
        .collect();
    tmp_new.sort_unstable();
    let tmp_new_len = tmp_new.len();
    tmp_new.dedup();
    // Если в матрице повторялись детали, то объединение невозможно
    if tmp_new.len() != tmp_new_len {
        return MatrixCompatibility::Incompatible;
    }
    // Матрицы совместимы
    MatrixCompatibility::Compatible(shared_shift_r, shared_shift_c, m_new_priority)
}

// Объединение матриц
fn merge_matrices(
    m_x: &Matrix,
    m_y: &Matrix,
    shared_shift_r: isize,
    shared_shift_c: isize,
) -> Matrix {
    // Размеры матриц x и y
    let (m_x_r, m_x_c) = (m_x.len(), m_x[0].len());
    let (m_y_r, m_y_c) = (m_y.len(), m_y[0].len());
    // Переход из системы координат матрицы x к с. к. новой матрицы
    let (d_r, d_c) = (
        max(0, shared_shift_r) as usize,
        max(0, shared_shift_c) as usize,
    );
    // Переход из системы координат матрицы y к с. к. новой матрицы
    let (d_minus_shift_r, d_minus_shift_c) = (
        ((d_r as isize) - shared_shift_r) as usize,
        ((d_c as isize) - shared_shift_c) as usize,
    );
    // Размер новой матрицы
    let (m_new_r, m_new_c) = (
        max(m_x_r + d_r, m_y_r + d_minus_shift_r),
        max(m_x_c + d_c, m_y_c + d_minus_shift_c),
    );

    // Новая матрица
    let mut m_new = vec![vec![(usize::MAX, usize::MAX); m_new_c]; m_new_r];
    // Копирование в неё матрицы x
    for (r_x, v_x) in m_x.iter().enumerate() {
        for (c_x, x) in v_x.iter().enumerate() {
            let (r_new, c_new) = (r_x + d_r, c_x + d_c);
            m_new[r_new][c_new] = *x;
        }
    }
    // Копирование в неё матрицы y
    for (r_y, v_y) in m_y.iter().enumerate() {
        for (c_y, y) in v_y.iter().enumerate() {
            let (r_new, c_new) = (r_y + d_minus_shift_r, c_y + d_minus_shift_c);
            m_new[r_new][c_new] = *y;
        }
    }
    m_new
}

fn merge_matrices_groups(
    img_width: usize,
    img_height: usize,
    pieces_compatibility: &[Vec<Vec<f32>>; 2],
    sl_all: Vec<Vec<Matrix>>,
) -> Vec<Matrix> {
    type AvailablePairsType = BTreeMap<MatrixPriority, Vec<(MatrixCompatibility, usize, usize)>>;

    let mut available_pairs: AvailablePairsType = BTreeMap::new();
    let mut used = Vec::new();
    let mut matrices_last = Vec::new();
    let add_new_matrix = |available_pairs: &mut AvailablePairsType,
                          used: &mut Vec<bool>,
                          matrices_last: &mut Vec<(Matrix, MatrixPriority)>,
                          m_x: Matrix,
                          check: bool| {
        let m_x_i = matrices_last.len();
        let m_x_priority = matrix_priority(img_width, pieces_compatibility, &m_x);
        if check {
            for (m_y_i, (m_y, m_y_priority)) in matrices_last.iter().enumerate() {
                if used[m_y_i] {
                    continue;
                }
                let order = m_x_priority < *m_y_priority;
                let can_merge = if order {
                    can_merge_matrices(img_width, img_height, pieces_compatibility, &m_x, m_y)
                } else {
                    can_merge_matrices(img_width, img_height, pieces_compatibility, m_y, &m_x)
                };

                let pairs_key = match can_merge {
                    MatrixCompatibility::Compatible(_, _, m_new_priority) => m_new_priority,
                    MatrixCompatibility::Incompatible => {
                        if order {
                            m_x_priority
                        } else {
                            *m_y_priority
                        }
                    }
                    MatrixCompatibility::NotRelated => continue,
                };

                let v = match available_pairs.get_mut(&pairs_key) {
                    Some(v) => v,
                    None => {
                        available_pairs.insert(pairs_key, Vec::new());
                        available_pairs.get_mut(&pairs_key).unwrap()
                    }
                };
                if order {
                    v.push((can_merge, m_x_i, m_y_i));
                } else {
                    v.push((can_merge, m_y_i, m_x_i));
                }
            }
        }
        used.push(false);
        matrices_last.push((m_x, m_x_priority));
    };

    for sl_curr in sl_all.into_iter().rev() {
        for sl in sl_curr.into_iter() {
            add_new_matrix(
                &mut available_pairs,
                &mut used,
                &mut matrices_last,
                sl,
                true,
            );
        }

        while !available_pairs.is_empty() {
            let (pairs_key, pairs_v) = available_pairs.iter_mut().next().unwrap();
            if pairs_v.is_empty() {
                let pairs_key = *pairs_key;
                available_pairs.remove(&pairs_key);
                continue;
            }
            let (m_comp, x_i, y_i) = pairs_v.pop().unwrap();
            if used[x_i] || used[y_i] {
                continue;
            }

            match m_comp {
                MatrixCompatibility::Compatible(shared_shift_r, shared_shift_c, _) => {
                    let m_new = merge_matrices(
                        &matrices_last[x_i].0,
                        &matrices_last[y_i].0,
                        shared_shift_r,
                        shared_shift_c,
                    );

                    used[x_i] = true;
                    used[y_i] = true;
                    add_new_matrix(
                        &mut available_pairs,
                        &mut used,
                        &mut matrices_last,
                        m_new,
                        true,
                    );
                }
                MatrixCompatibility::Incompatible => {
                    used[y_i] = true;
                }
                _ => unreachable!(),
            }
        }

        let matrices_next: Vec<_> = matrices_last
            .iter()
            .enumerate()
            .filter_map(|(i, m)| if used[i] { None } else { Some(m.0.clone()) })
            .collect();
        available_pairs.clear();
        used.clear();
        matrices_last.clear();
        for m in matrices_next {
            add_new_matrix(
                &mut available_pairs,
                &mut used,
                &mut matrices_last,
                m,
                false,
            );
        }
    }

    matrices_last.sort_by_key(|m| m.1);
    matrices_last.into_iter().map(|m| m.0).collect()
}

fn fill_greedy(
    img_width: usize,
    img_height: usize,
    pieces_compatibility: &[Vec<Vec<f32>>; 2],
    solution: &mut Solution,
) {
    let mut free_pieces: IndexSet<_, FxBuildHasher> = (0..img_height)
        .flat_map(|r| (0..img_width).map(move |c| (r, c)))
        .collect();
    for piece in solution.iter() {
        free_pieces.remove(piece);
    }

    let mut free_positions: [IndexSet<(usize, usize), FxBuildHasher>; 5] = [
        IndexSet::with_hasher(FxBuildHasher::default()),
        IndexSet::with_hasher(FxBuildHasher::default()),
        IndexSet::with_hasher(FxBuildHasher::default()),
        IndexSet::with_hasher(FxBuildHasher::default()),
        IndexSet::with_hasher(FxBuildHasher::default()),
    ];
    let add_to_free_positions = |solution: &mut Solution,
                                 free_positions: &mut [IndexSet<(usize, usize), FxBuildHasher>;
                                          5],
                                 r: usize,
                                 c: usize| {
        if solution[r * img_width + c].0 != usize::MAX {
            return;
        }
        let mut cnt = 0;
        for dr in [-1isize, 1] {
            for dc in [-1isize, 1] {
                let (new_r, new_c) = ((r as isize) + dr, (c as isize) + dc);
                if new_r < 0
                    || new_r >= (img_height as isize)
                    || new_c < 0
                    || new_c >= (img_width as isize)
                {
                    continue;
                }
                let (new_r, new_c) = (new_r as usize, new_c as usize);
                if solution[new_r * img_width + new_c].0 == usize::MAX {
                    cnt += 1;
                }
            }
        }
        free_positions[cnt].insert((r, c));
    };
    for r in 0..img_height {
        for c in 0..img_width {
            add_to_free_positions(solution, &mut free_positions, r, c);
        }
    }

    while !free_pieces.is_empty() {
        for cnt in 0..=4 {
            if free_positions[cnt].is_empty() {
                continue;
            }
            let (pos_r, pos_c) = *free_positions[cnt].first().unwrap();

            // Деталь слева
            let (left_piece_r, left_piece_c) = if pos_c == 0 {
                (usize::MAX, usize::MAX)
            } else {
                solution[pos_r * img_width + pos_c - 1]
            };
            // Деталь справа
            let (right_piece_r, right_piece_c) = if pos_c == img_width - 1 {
                (usize::MAX, usize::MAX)
            } else {
                solution[pos_r * img_width + pos_c + 1]
            };
            // Деталь сверху
            let (up_piece_r, up_piece_c) = if pos_r == 0 {
                (usize::MAX, usize::MAX)
            } else {
                solution[(pos_r - 1) * img_width + pos_c]
            };
            // Деталь снизу
            let (down_piece_r, down_piece_c) = if pos_r == img_height - 1 {
                (usize::MAX, usize::MAX)
            } else {
                solution[(pos_r + 1) * img_width + pos_c]
            };

            // Наименьшая несовместимость
            let mut best_compatibility = f32::INFINITY;
            // Наиболее подходящая деталь
            let mut best_piece = (usize::MAX, usize::MAX);
            // Обработка всех свободных деталей
            for (piece_r, piece_c) in &free_pieces {
                // Несовместимость
                let mut res = 0.0f32;
                // Обработка детали слева
                if left_piece_r != usize::MAX {
                    res += pieces_compatibility[0][left_piece_r * img_width + left_piece_c]
                        [piece_r * img_width + piece_c];
                    if res >= best_compatibility {
                        continue;
                    }
                }
                // Обработка детали справа
                if right_piece_r != usize::MAX {
                    res += pieces_compatibility[0][piece_r * img_width + piece_c]
                        [right_piece_r * img_width + right_piece_c];
                    if res >= best_compatibility {
                        continue;
                    }
                }
                // Обработка детали сверху
                if up_piece_r != usize::MAX {
                    res += pieces_compatibility[1][up_piece_r * img_width + up_piece_c]
                        [piece_r * img_width + piece_c];
                    if res >= best_compatibility {
                        continue;
                    }
                }
                // Обработка детали снизу
                if down_piece_r != usize::MAX {
                    res += pieces_compatibility[1][piece_r * img_width + piece_c]
                        [down_piece_r * img_width + down_piece_c];
                    if res >= best_compatibility {
                        continue;
                    }
                }

                // Обновление наименее несовместимой детали
                best_piece = (*piece_r, *piece_c);
                best_compatibility = res;
            }

            free_pieces.remove(&best_piece);
            free_positions[cnt].remove(&(pos_r, pos_c));
            solution[pos_r * img_width + pos_c] = best_piece;

            if pos_c != 0 {
                for cnt_adj in 0..=4 {
                    if free_positions[cnt_adj].remove(&(pos_r, pos_c - 1)) {
                        add_to_free_positions(solution, &mut free_positions, pos_r, pos_c - 1);
                        break;
                    }
                }
            }
            if pos_c != img_width - 1 {
                for cnt_adj in 0..=4 {
                    if free_positions[cnt_adj].remove(&(pos_r, pos_c + 1)) {
                        add_to_free_positions(solution, &mut free_positions, pos_r, pos_c + 1);
                        break;
                    }
                }
            }
            if pos_r != 0 {
                for cnt_adj in 0..=4 {
                    if free_positions[cnt_adj].remove(&(pos_r - 1, pos_c)) {
                        add_to_free_positions(solution, &mut free_positions, pos_r - 1, pos_c);
                        break;
                    }
                }
            }
            if pos_r != img_height - 1 {
                for cnt_adj in 0..=4 {
                    if free_positions[cnt_adj].remove(&(pos_r + 1, pos_c)) {
                        add_to_free_positions(solution, &mut free_positions, pos_r + 1, pos_c);
                        break;
                    }
                }
            }

            break;
        }
    }
}

// Шаг алгоритма
pub fn algorithm_step(
    img_width: usize,
    img_height: usize,
    pieces_compatibility: &[Vec<Vec<f32>>; 2],
    pieces_match_candidates: &[Vec<Vec<(usize, usize)>>; 2],
) -> Vec<Solution> {
    let sl_all = small_loops(img_width, img_height, pieces_match_candidates);
    println!("{:?}", sl_all.iter().map(|v| v.len()).collect::<Vec<_>>());
    let matrices_last = merge_matrices_groups(img_width, img_height, pieces_compatibility, sl_all);
    let matrices_last_first = matrices_last.first().unwrap();
    let mut solution = vec![(usize::MAX, usize::MAX); img_width * img_height];
    for r in 0..matrices_last_first.len() {
        for c in 0..matrices_last_first[r].len() {
            if matrices_last_first[r][c].0 != usize::MAX {
                solution[r * img_width + c] = matrices_last_first[r][c];
            }
        }
    }
    fill_greedy(img_width, img_height, pieces_compatibility, &mut solution);
    vec![solution]
}
