use image::RgbaImage;
use image_processing::get_lab_image;
use rand::{prelude::SliceRandom, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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

// Вычисление несовместимостей деталей
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

    // Несовместимость пикселей - разность цветов в пространстве L*a*b*
    let two_pixels_dissimilarity =
        |i: usize, j: usize| lab_pixels[i].squared_distance(&lab_pixels[j]);

    // Вычисление несовместимости детали i (строка i_r, столбец i_c),
    // находящейся слева от детали j (строка j_r, столбец j_c)
    let right_dissimilarity: Vec<Vec<f32>> = (0..img_height)
        .into_par_iter()
        .flat_map_iter(|i_r| {
            (0..img_width).map(move |i_c| {
                (0..img_height)
                    .flat_map(|j_r| {
                        (0..img_width).map(move |j_c| {
                            (0..piece_size)
                                .map(|k| {
                                    two_pixels_dissimilarity(
                                        (i_r * piece_size + k) * image_width
                                            + i_c * piece_size
                                            + piece_size
                                            - 1,
                                        (j_r * piece_size + k) * image_width + j_c * piece_size,
                                    )
                                })
                                .sum::<f32>()
                                .sqrt()
                        })
                    })
                    .collect()
            })
        })
        .collect();
    // Вычисление несовместимости детали i (строка i_r, столбец i_c),
    // находящейся сверху от детали j (строка j_r, столбец j_c)
    let down_dissimilarity: Vec<Vec<f32>> = (0..img_height)
        .into_par_iter()
        .flat_map_iter(|i_r| {
            (0..img_width).map(move |i_c| {
                (0..img_height)
                    .flat_map(|j_r| {
                        (0..img_width).map(move |j_c| {
                            (0..piece_size)
                                .map(|k| {
                                    two_pixels_dissimilarity(
                                        (i_r * piece_size + piece_size - 1) * image_width
                                            + i_c * piece_size
                                            + k,
                                        j_r * piece_size * image_width + j_c * piece_size + k,
                                    )
                                })
                                .sum::<f32>()
                                .sqrt()
                        })
                    })
                    .collect()
            })
        })
        .collect();
    [right_dissimilarity, down_dissimilarity]
}

// Оценка решения
fn chromosome_dissimilarity(
    img_width: usize,
    img_height: usize,
    pieces_dissimilarity: &[Vec<Vec<f32>>; 2],
    solution: &Solution,
) -> f32 {
    // Несовместимости деталей по направлению вправо
    let right_dissimilarities = (0..img_height)
        .map(|i_r| {
            (0..(img_width - 1))
                .map(|i_c| {
                    let (j_r, j_c) = solution[i_r * img_width + i_c];
                    let (k_r, k_c) = solution[i_r * img_width + i_c + 1];
                    pieces_dissimilarity[0][j_r * img_width + j_c][k_r * img_width + k_c]
                })
                .sum::<f32>()
        })
        .sum::<f32>();
    // Несовместимости деталей по направлению вниз
    let down_dissimilarities = (0..(img_height - 1))
        .map(|i_r| {
            (0..img_width)
                .map(|i_c| {
                    let (j_r, j_c) = solution[i_r * img_width + i_c];
                    let (k_r, k_c) = solution[(i_r + 1) * img_width + i_c];
                    pieces_dissimilarity[1][j_r * img_width + j_c][k_r * img_width + k_c]
                })
                .sum::<f32>()
        })
        .sum::<f32>();
    right_dissimilarities + down_dissimilarities
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
