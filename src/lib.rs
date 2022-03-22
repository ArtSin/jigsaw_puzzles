use image::RgbaImage;
use image_processing::get_lab_image;
use rand::{prelude::SliceRandom, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::image_processing::get_rgb_image;

pub mod genetic_algorithm;
pub mod image_processing;

// Решение пазла: двумерный (представлен как одномерный, записан построчно) массив,
// в каждой позиции которого находится деталь (пара из строки и столбца)
pub type Solution = Vec<(usize, usize)>;

// Направления возможных соседей: верх, низ, лево, право
const NEIGHBOUR_DIRECTIONS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

// Создание случайного решения
fn generate_random_solution<T: Rng>(img_width: usize, img_height: usize, rng: &mut T) -> Solution {
    let mut rand_nums: Vec<_> = (0..img_height)
        .flat_map(|r| (0..img_width).map(move |c| (r, c)))
        .collect();
    rand_nums.shuffle(rng);
    rand_nums
}

// Вычисление "несходства" деталей
pub fn calculate_dissimilarities(
    image: &RgbaImage,
    img_width: usize,
    img_height: usize,
    piece_size: usize,
) -> [Vec<Vec<f32>>; 2] {
    // Пиксели изображения в цветовом пространстве L*a*b*
    let lab_pixels = get_lab_image(image);
    // Ширина изображения в пикселях
    let image_width = img_width * piece_size;

    // Несходство пикселей - разность цветов в пространстве L*a*b*
    let two_pixels_dissimilarity =
        |i: usize, j: usize| lab_pixels[i].squared_distance(&lab_pixels[j]);

    // Вычисление несходства детали i (строка i_r, столбец i_c),
    // находящейся слева от детали j (строка j_r, столбец j_c)
    let right_dissimilarity: Vec<Vec<f32>> = (0..img_height)
        .into_par_iter()
        .flat_map_iter(|i_r| {
            (0..img_width)
                .map(|i_c| {
                    (0..img_height)
                        .flat_map(|j_r| {
                            (0..img_width)
                                .map(|j_c| {
                                    (0..piece_size)
                                        .map(|k| {
                                            two_pixels_dissimilarity(
                                                // k-й элемент последнего столбца i-й детали
                                                (i_r * piece_size + k) * image_width
                                                    + i_c * piece_size
                                                    + piece_size
                                                    - 1,
                                                // k-й элемент первого столбца j-й детали
                                                (j_r * piece_size + k) * image_width
                                                    + j_c * piece_size,
                                            )
                                        })
                                        .sum::<f32>()
                                        .sqrt()
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    // Вычисление несходства детали i (строка i_r, столбец i_c),
    // находящейся сверху от детали j (строка j_r, столбец j_c)
    let down_dissimilarity: Vec<Vec<f32>> = (0..img_height)
        .into_par_iter()
        .flat_map_iter(|i_r| {
            (0..img_width)
                .map(|i_c| {
                    (0..img_height)
                        .flat_map(|j_r| {
                            (0..img_width)
                                .map(|j_c| {
                                    (0..piece_size)
                                        .map(|k| {
                                            two_pixels_dissimilarity(
                                                // k-й элемент последней строки i-й детали
                                                (i_r * piece_size + piece_size - 1) * image_width
                                                    + i_c * piece_size
                                                    + k,
                                                // k-й элемент первой строки j-й детали
                                                j_r * piece_size * image_width
                                                    + j_c * piece_size
                                                    + k,
                                            )
                                        })
                                        .sum::<f32>()
                                        .sqrt()
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    [right_dissimilarity, down_dissimilarity]
}

// Вычисление MGC (Mahalanobis Gradient Compatibility) для деталей
pub fn calculate_mgc(
    image: &RgbaImage,
    img_width: usize,
    img_height: usize,
    piece_size: usize,
) -> [Vec<Vec<f32>>; 2] {
    // Пиксели изображения в цветовом пространстве RGB
    let rgb_pixels = get_rgb_image(image);
    // Ширина изображения в пикселях
    let image_width = img_width * piece_size;

    // Вычисление MGC для одной стороны (по градиентам в одной из деталей и градиенту между ними)
    let calc_mgc_part = |grad_side: Vec<[f32; 3]>, grad_mid: Vec<[f32; 3]>| {
        const EPS: f32 = 1e-6;

        let sz = grad_side.len() as f32;
        // Средний градиент по каждому цветовому каналу
        let means = [0, 1, 2].map(|i| grad_side.iter().map(|v| v[i]).sum::<f32>() / sz);
        // Коэффициент ковариации
        let cov_denom = 1.0 / (sz - 1.0);
        // Вычисление ковариации для разных величин
        let get_cov = |i, j| {
            grad_side
                .iter()
                .map(|v| (v[i] - means[i]) * (v[j] - means[j]))
                .sum::<f32>()
                * cov_denom
        };
        // Вычисление ковариации величины с самой собой (дисперсии)
        let get_cov_sqr = |i: usize| {
            grad_side
                .iter()
                .map(|v| (v[i] - means[i]).powi(2))
                .sum::<f32>()
                * cov_denom
        };
        // Матрица ковариаций между градиентами по каждому цветовому каналу:
        // [[a b c]]
        // [[d e f]]
        // [[g h i]]
        // Так как она симметрична, то d = b, g = c, h = f, и они не вычисляются
        // К элементам на диагонали добавляется EPS для численной стабильности инвертирования матрицы
        let a = get_cov_sqr(0) + EPS;
        let e = get_cov_sqr(1) + EPS;
        let i = get_cov_sqr(2) + EPS;
        let b = get_cov(0, 1);
        let c = get_cov(0, 2);
        let f = get_cov(1, 2);
        // Вычисление обратной к матрице ковариаций:
        //               1                    [[(ei - fh) (ch - bi) (bf - ce)]]
        // ---------------------------------  [[(fg - di) (ai - cg) (cd - af)]]
        // a * ei_ff + b * cf_bi + c * bf_ce  [[(dh - eg) (bg - ah) (ae - bd)]]
        // Элементы матрицы (d, g, h заменены)
        let ei_ff = e * i - f * f;
        let cf_bi = c * f - b * i;
        let bf_ce = b * f - c * e;
        let bc_af = b * c - a * f;
        let ai_cc = a * i - c * c;
        let ae_bb = a * e - b * b;
        // Коэффициент обратной матрицы
        let denom = 1.0 / (a * ei_ff + b * cf_bi + c * bf_ce);
        // Умножение на коэффициент и на 2 (при необходимости для следующих формул)
        let ei_ff = ei_ff * denom;
        let cf_bi = 2.0 * (cf_bi * denom);
        let bf_ce = 2.0 * (bf_ce * denom);
        let bc_af = 2.0 * (bc_af * denom);
        let ai_cc = ai_cc * denom;
        let ae_bb = ae_bb * denom;
        // Разность между градиентом из одной детали в другую и средним градиентом в детали
        let grad_diff: Vec<_> = grad_mid
            .iter()
            .map(|v| [v[0] - means[0], v[1] - means[1], v[2] - means[2]])
            .collect();
        // Вычисление результата (. - умножение матриц):
        // sz
        //  ∑ grad_diff[i] . covariations_inv . (grad_diff[i])^T
        // i=0
        // Можно преобразовать как сумму элементов матрицы вида (* - поточечное умножение):
        // grad_diff . covariations_inv * grad_diff
        // Эта формула вычисляется построчно для grad_diff и подставляются элементы матрицы,
        // обратной к матрице ковариаций
        let res = grad_diff
            .iter()
            .map(|v| {
                (v[0] * ei_ff + v[1] * cf_bi + v[2] * bf_ce) * v[0]
                    + (v[1] * ai_cc + v[2] * bc_af) * v[1]
                    + ae_bb * v[2] * v[2]
            })
            .sum::<f32>();
        f32::max(0.0, res).sqrt()
    };

    // Вычисление MGC для детали i (строка i_r, столбец i_c),
    // находящейся слева от детали j (строка j_r, столбец j_c)
    let right_mgc: Vec<Vec<f32>> = (0..img_height)
        .into_par_iter()
        .flat_map_iter(|i_r| {
            (0..img_width)
                .map(|i_c| {
                    (0..img_height)
                        .flat_map(|j_r| {
                            (0..img_width)
                                .map(|j_c| {
                                    // Градиент из предпоследнего столбца i-й детали в её последний столбец
                                    let grad_l = (0..piece_size)
                                        .map(|k| {
                                            [0, 1, 2].map(|c| {
                                                (rgb_pixels[(i_r * piece_size + k) * image_width
                                                    + i_c * piece_size
                                                    + piece_size
                                                    - 1][c]
                                                    - rgb_pixels[(i_r * piece_size + k)
                                                        * image_width
                                                        + i_c * piece_size
                                                        + piece_size
                                                        - 2][c])
                                                    as f32
                                            })
                                        })
                                        .collect::<Vec<_>>();
                                    // Градиент из последнего столбца i-й детали в первый столбец j-й детали
                                    let grad_lr = (0..piece_size)
                                        .map(|k| {
                                            [0, 1, 2].map(|c| {
                                                (rgb_pixels[(j_r * piece_size + k) * image_width
                                                    + j_c * piece_size][c]
                                                    - rgb_pixels[(i_r * piece_size + k)
                                                        * image_width
                                                        + i_c * piece_size
                                                        + piece_size
                                                        - 1][c])
                                                    as f32
                                            })
                                        })
                                        .collect::<Vec<_>>();
                                    // Градиент из первого столбца j-й детали в последний столбец i-й детали
                                    let grad_rl = grad_lr
                                        .iter()
                                        .map(|x: &[f32; 3]| [-x[0], -x[1], -x[2]])
                                        .collect::<Vec<_>>();
                                    // Градиент из второго столбца j-й детали в её первый столбец
                                    let grad_r = (0..piece_size)
                                        .map(|k| {
                                            [0, 1, 2].map(|c| {
                                                (rgb_pixels[(j_r * piece_size + k) * image_width
                                                    + j_c * piece_size][c]
                                                    - rgb_pixels[(j_r * piece_size + k)
                                                        * image_width
                                                        + j_c * piece_size
                                                        + 1][c])
                                                    as f32
                                            })
                                        })
                                        .collect::<Vec<_>>();
                                    // Вычисление MGC как суммы MGC слева направо и справа налево
                                    calc_mgc_part(grad_l, grad_lr) + calc_mgc_part(grad_r, grad_rl)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    // Вычисление MGC для детали i (строка i_r, столбец i_c),
    // находящейся сверху от детали j (строка j_r, столбец j_c)
    let down_mgc: Vec<Vec<f32>> = (0..img_height)
        .into_par_iter()
        .flat_map_iter(|i_r| {
            (0..img_width)
                .map(|i_c| {
                    (0..img_height)
                        .flat_map(|j_r| {
                            (0..img_width)
                                .map(|j_c| {
                                    // Градиент из предпоследней строки i-й детали в её последнюю строку
                                    let grad_u = (0..piece_size)
                                        .map(|k| {
                                            [0, 1, 2].map(|c| {
                                                (rgb_pixels[(i_r * piece_size + piece_size - 1)
                                                    * image_width
                                                    + i_c * piece_size
                                                    + k][c]
                                                    - rgb_pixels[(i_r * piece_size + piece_size
                                                        - 2)
                                                        * image_width
                                                        + i_c * piece_size
                                                        + k][c])
                                                    as f32
                                            })
                                        })
                                        .collect::<Vec<_>>();
                                    // Градиент из последней строки i-й детали в первую строку j-й детали
                                    let grad_ud = (0..piece_size)
                                        .map(|k| {
                                            [0, 1, 2].map(|c| {
                                                (rgb_pixels[j_r * piece_size * image_width
                                                    + j_c * piece_size
                                                    + k][c]
                                                    - rgb_pixels[(i_r * piece_size + piece_size
                                                        - 1)
                                                        * image_width
                                                        + i_c * piece_size
                                                        + k][c])
                                                    as f32
                                            })
                                        })
                                        .collect::<Vec<_>>();
                                    // Градиент из первого столбца j-й детали в последний столбец i-й детали
                                    let grad_du = grad_ud
                                        .iter()
                                        .map(|x: &[f32; 3]| [-x[0], -x[1], -x[2]])
                                        .collect::<Vec<_>>();
                                    // Градиент из второй строки j-й детали в её первую строку
                                    let grad_d = (0..piece_size)
                                        .map(|k| {
                                            [0, 1, 2].map(|c| {
                                                (rgb_pixels[j_r * piece_size * image_width
                                                    + j_c * piece_size
                                                    + k][c]
                                                    - rgb_pixels[(j_r * piece_size + 1)
                                                        * image_width
                                                        + j_c * piece_size
                                                        + k][c])
                                                    as f32
                                            })
                                        })
                                        .collect::<Vec<_>>();
                                    // Вычисление MGC как суммы MGC сверху вниз и снизу вверх
                                    calc_mgc_part(grad_u, grad_ud) + calc_mgc_part(grad_d, grad_du)
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    [right_mgc, down_mgc]
}

// Оценка решения
fn solution_compatibility(
    img_width: usize,
    img_height: usize,
    pieces_compatibility: &[Vec<Vec<f32>>; 2],
    solution: &Solution,
) -> f32 {
    // Совместимости деталей по направлению вправо
    let right_compatibility = (0..img_height)
        .map(|i_r| {
            (0..(img_width - 1))
                .map(|i_c| {
                    let (j_r, j_c) = solution[i_r * img_width + i_c];
                    let (k_r, k_c) = solution[i_r * img_width + i_c + 1];
                    pieces_compatibility[0][j_r * img_width + j_c][k_r * img_width + k_c]
                })
                .sum::<f32>()
        })
        .sum::<f32>();
    // Совместимости деталей по направлению вниз
    let down_compatibility = (0..(img_height - 1))
        .map(|i_r| {
            (0..img_width)
                .map(|i_c| {
                    let (j_r, j_c) = solution[i_r * img_width + i_c];
                    let (k_r, k_c) = solution[(i_r + 1) * img_width + i_c];
                    pieces_compatibility[1][j_r * img_width + j_c][k_r * img_width + k_c]
                })
                .sum::<f32>()
        })
        .sum::<f32>();
    right_compatibility + down_compatibility
}

// Прямое сравнение: соотношение числа деталей в правильных позициях к числу всех деталей
pub fn image_direct_comparison(img_width: usize, solution: &Solution) -> f32 {
    (solution
        .iter()
        .enumerate()
        .filter(|(rc, ij)| *rc == ij.0 * img_width + ij.1)
        .count() as f32)
        * 100.0
        / (solution.len() as f32)
}

// Сравнение по соседям: соотношение числа правильных соседей к числу соседей, среднее по всем деталям
pub fn image_neighbour_comparison(img_width: usize, img_height: usize, solution: &Solution) -> f32 {
    (0..img_height)
        .map(|r| {
            (0..img_width)
                .map(|c| {
                    // Число правильных соседей, число всех соседей
                    let (mut res, mut count) = (0usize, 0usize);
                    // Текущая деталь
                    let (i, j) = solution[r * img_width + c];
                    // Перебор соседей
                    for (dr, dc) in NEIGHBOUR_DIRECTIONS {
                        // Сосед позиции
                        let (new_r, new_c) = ((r as isize) + dr, (c as isize) + dc);
                        // Нет соседа, так как позиция на грани изображения
                        if new_r < 0 || new_c < 0 {
                            continue;
                        }
                        let (new_r, new_c) = (new_r as usize, new_c as usize);
                        if new_r >= img_height || new_c >= img_width {
                            continue;
                        }
                        // Сосед есть
                        count += 1;

                        // Сосед детали в исходном изображении
                        let (new_i, new_j) = ((i as isize) + dr, (j as isize) + dc);
                        // Нет соседа
                        if new_i < 0 || new_j < 0 {
                            continue;
                        }
                        let (new_i, new_j) = (new_i as usize, new_j as usize);
                        if new_i >= img_height || new_j >= img_width {
                            continue;
                        }

                        // Если сосед по позиции оказался соседом детали в исходном изображении, то он правильный
                        if solution[new_r * img_width + new_c] == (new_i, new_j) {
                            res += 1;
                        }
                    }
                    (res as f32) / (count as f32)
                })
                .sum::<f32>()
        })
        .sum::<f32>()
        * 100.0
        / (solution.len() as f32)
}
