use std::{
    cmp::{max, min},
    collections::BTreeSet,
};

use float_ord::FloatOrd;
use image::RgbaImage;
use rand::{prelude::IteratorRandom, seq::SliceRandom, Rng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::image_processing::get_lab_image;

pub type Chromosome = Vec<Vec<(usize, usize)>>;

const ELITISM_COUNT: usize = 4;

// Вычисление совместимостей деталей
pub fn calculate_dissimilarities(
    image: &RgbaImage,
    img_width: usize,
    img_height: usize,
    piece_size: usize,
) -> [Vec<Vec<f32>>; 2] {
    let lab_pixels = get_lab_image(image);
    let image_width = img_width * piece_size;

    let two_pixels_dissimilarity = |i: usize, j: usize| {
        (lab_pixels[i].l - lab_pixels[j].l).powi(2)
            + (lab_pixels[i].a - lab_pixels[j].a).powi(2)
            + (lab_pixels[i].b - lab_pixels[j].b).powi(2)
    };

    // Деталь i (строка i_r, столбец i_c) слева от детали j (строка j_r, столбец j_c)
    let right_dissimilarity: Vec<Vec<f32>> = (0..img_height)
        .flat_map(|i_r| {
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
    // Деталь i (строка i_r, столбец i_c) сверху от детали j (строка j_r, столбец j_c)
    let down_dissimilarity: Vec<Vec<f32>> = (0..img_height)
        .flat_map(|i_r| {
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

pub fn find_best_buddies(
    img_width: usize,
    img_height: usize,
    pieces_dissimilarity: &[Vec<Vec<f32>>; 2],
) -> [Vec<(usize, usize)>; 4] {
    let get_buddies = |i: usize| {
        pieces_dissimilarity[i]
            .iter()
            .map(|v| {
                v.iter()
                    .enumerate()
                    .reduce(|x, y| if x.1 <= y.1 { x } else { y })
                    .map(|(i, _)| (i / img_width, i % img_width))
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let right_buddies = get_buddies(0);
    let down_buddies = get_buddies(1);

    let get_reverse_buddies = |buddies: &[(usize, usize)]| {
        let mut res = vec![(usize::MAX, usize::MAX); buddies.len()];
        for r in 0..img_height {
            for c in 0..img_width {
                let (i, j) = buddies[r * img_width + c];
                res[i * img_width + j] = (r, c);
            }
        }
        res
    };
    let left_buddies = get_reverse_buddies(&right_buddies);
    let up_buddies = get_reverse_buddies(&down_buddies);

    [right_buddies, down_buddies, left_buddies, up_buddies]
}

// Оценка хромосомы
fn chromosome_dissimilarity(
    img_width: usize,
    img_height: usize,
    pieces_dissimilarity: &[Vec<Vec<f32>>; 2],
    chromosome: &Chromosome,
) -> f32 {
    let right_dissimilarities = (0..img_height)
        .map(|i_r| {
            (0..(img_width - 1))
                .map(|i_c| {
                    let (j_r, j_c) = chromosome[i_r][i_c];
                    let (k_r, k_c) = chromosome[i_r][i_c + 1];
                    pieces_dissimilarity[0][j_r * img_width + j_c][k_r * img_width + k_c]
                })
                .sum::<f32>()
        })
        .sum::<f32>();
    let down_dissimilarities = (0..(img_height - 1))
        .map(|i_r| {
            (0..img_width)
                .map(|i_c| {
                    let (j_r, j_c) = chromosome[i_r][i_c];
                    let (k_r, k_c) = chromosome[i_r + 1][i_c];
                    pieces_dissimilarity[1][j_r * img_width + j_c][k_r * img_width + k_c]
                })
                .sum::<f32>()
        })
        .sum::<f32>();
    right_dissimilarities + down_dissimilarities
}

// Создание случайной хромосомы
fn generate_random_chromosome<T: Rng>(
    img_width: usize,
    img_height: usize,
    rng: &mut T,
) -> Chromosome {
    let mut rand_nums: Vec<_> = (0..img_height)
        .flat_map(|r| (0..img_width).map(move |c| (r, c)))
        .collect();
    rand_nums.shuffle(rng);
    rand_nums.chunks(img_width).map(|v| v.to_vec()).collect()
}

// Скрещивание
fn chromosomes_crossover(
    img_width: usize,
    img_height: usize,
    pieces_dissimilarity: &[Vec<Vec<f32>>; 2],
    pieces_buddies: &[Vec<(usize, usize)>; 4],
    chromosome_1: &Chromosome,
    chromosome_2: &Chromosome,
    mut rng: Xoshiro256PlusPlus,
) -> Chromosome {
    let start_piece = (
        (0..img_height).choose(&mut rng).unwrap(),
        (0..img_width).choose(&mut rng).unwrap(),
    );

    let mut pos_in_chromosome_1 = vec![vec![(usize::MAX, usize::MAX); img_width]; img_height];
    let mut pos_in_chromosome_2 = pos_in_chromosome_1.clone();
    for r in 0..img_height {
        for c in 0..img_width {
            let (i, j) = chromosome_1[r][c];
            pos_in_chromosome_1[i][j] = (r, c);

            let (i, j) = chromosome_2[r][c];
            pos_in_chromosome_2[i][j] = (r, c);
        }
    }

    let mut free_pieces: BTreeSet<_> = (0..img_height)
        .flat_map(|r| (0..img_width).map(move |c| (r, c)))
        .collect();
    let mut free_positions = Vec::new();
    let mut tmp_chromosome: Chromosome = (0..(2 * img_height))
        .map(|_| {
            (0..(2 * img_width))
                .map(|_| (usize::MAX, usize::MAX))
                .collect()
        })
        .collect();

    let mut min_r = img_height;
    let mut max_r = img_height;
    let mut min_c = img_width;
    let mut max_c = img_width;
    free_positions.push((img_height, img_width));
    while !free_positions.is_empty() {
        let mut selected_pos = None;
        let mut selected_piece = None;

        // Первая деталь
        if free_positions.len() == 1 && free_positions[0] == (img_height, img_width) {
            selected_pos = Some((img_height, img_width));
            selected_piece = Some(start_piece);
        }

        // Удаление неправильных позиций
        free_positions.retain(|(r, c)| {
            1 + max(max_r, *r) - min(min_r, *r) <= img_height
                && 1 + max(max_c, *c) - min(min_c, *c) <= img_width
        });
        if free_positions.is_empty() {
            break;
        }

        // Phase 1 (both parents agree)
        if selected_pos.is_none() {
            if let Some((pos, piece)) = free_positions
                .iter()
                .filter_map(|(pos_r, pos_c)| {
                    // Деталь слева
                    if *pos_c != 0 {
                        let (left_piece_r, left_piece_c) = tmp_chromosome[*pos_r][*pos_c - 1];
                        if left_piece_r != usize::MAX {
                            let (pos_1_r, pos_1_c) =
                                pos_in_chromosome_1[left_piece_r][left_piece_c];
                            let (pos_2_r, pos_2_c) =
                                pos_in_chromosome_2[left_piece_r][left_piece_c];
                            if pos_1_c != img_width - 1
                                && pos_2_c != img_width - 1
                                && chromosome_1[pos_1_r][pos_1_c + 1]
                                    == chromosome_2[pos_2_r][pos_2_c + 1]
                                && free_pieces.contains(&chromosome_1[pos_1_r][pos_1_c + 1])
                            {
                                return Some((
                                    (*pos_r, *pos_c),
                                    chromosome_1[pos_1_r][pos_1_c + 1],
                                ));
                            }
                        }
                    }
                    // Деталь справа
                    if *pos_c != 2 * img_width - 1 {
                        let (right_piece_r, right_piece_c) = tmp_chromosome[*pos_r][*pos_c + 1];
                        if right_piece_r != usize::MAX {
                            let (pos_1_r, pos_1_c) =
                                pos_in_chromosome_1[right_piece_r][right_piece_c];
                            let (pos_2_r, pos_2_c) =
                                pos_in_chromosome_2[right_piece_r][right_piece_c];
                            if pos_1_c != 0
                                && pos_2_c != 0
                                && chromosome_1[pos_1_r][pos_1_c - 1]
                                    == chromosome_2[pos_2_r][pos_2_c - 1]
                                && free_pieces.contains(&chromosome_1[pos_1_r][pos_1_c - 1])
                            {
                                return Some((
                                    (*pos_r, *pos_c),
                                    chromosome_1[pos_1_r][pos_1_c - 1],
                                ));
                            }
                        }
                    }
                    // Деталь сверху
                    if *pos_r != 0 {
                        let (up_piece_r, up_piece_c) = tmp_chromosome[*pos_r - 1][*pos_c];
                        if up_piece_r != usize::MAX {
                            let (pos_1_r, pos_1_c) = pos_in_chromosome_1[up_piece_r][up_piece_c];
                            let (pos_2_r, pos_2_c) = pos_in_chromosome_2[up_piece_r][up_piece_c];
                            if pos_1_r != img_height - 1
                                && pos_2_r != img_height - 1
                                && chromosome_1[pos_1_r + 1][pos_1_c]
                                    == chromosome_2[pos_2_r + 1][pos_2_c]
                                && free_pieces.contains(&chromosome_1[pos_1_r + 1][pos_1_c])
                            {
                                return Some((
                                    (*pos_r, *pos_c),
                                    chromosome_1[pos_1_r + 1][pos_1_c],
                                ));
                            }
                        }
                    }
                    // Деталь снизу
                    if *pos_r != 2 * img_height - 1 {
                        let (down_piece_r, down_piece_c) = tmp_chromosome[*pos_r + 1][*pos_c];
                        if down_piece_r != usize::MAX {
                            let (pos_1_r, pos_1_c) =
                                pos_in_chromosome_1[down_piece_r][down_piece_c];
                            let (pos_2_r, pos_2_c) =
                                pos_in_chromosome_2[down_piece_r][down_piece_c];
                            if pos_1_r != 0
                                && pos_2_r != 0
                                && chromosome_1[pos_1_r - 1][pos_1_c]
                                    == chromosome_2[pos_2_r - 1][pos_2_c]
                                && free_pieces.contains(&chromosome_1[pos_1_r - 1][pos_1_c])
                            {
                                return Some((
                                    (*pos_r, *pos_c),
                                    chromosome_1[pos_1_r - 1][pos_1_c],
                                ));
                            }
                        }
                    }
                    None
                })
                .choose(&mut rng)
            {
                selected_pos = Some(pos);
                selected_piece = Some(piece);
            }
        }

        // Phase 2 (best-buddy)
        if selected_pos.is_none() {
            if let Some((pos, piece)) = free_positions
                .iter()
                .filter_map(|(pos_r, pos_c)| {
                    // Деталь слева
                    if *pos_c != 0 {
                        let (left_piece_r, left_piece_c) = tmp_chromosome[*pos_r][*pos_c - 1];
                        if left_piece_r != usize::MAX {
                            let best_buddy =
                                pieces_buddies[0][left_piece_r * img_width + left_piece_c];
                            let (pos_1_r, pos_1_c) =
                                pos_in_chromosome_1[left_piece_r][left_piece_c];
                            let (pos_2_r, pos_2_c) =
                                pos_in_chromosome_2[left_piece_r][left_piece_c];
                            if ((pos_1_c != img_width - 1
                                && chromosome_1[pos_1_r][pos_1_c + 1] == best_buddy)
                                || (pos_2_c != img_width - 1
                                    && chromosome_2[pos_2_r][pos_2_c + 1] == best_buddy))
                                && free_pieces.contains(&best_buddy)
                            {
                                return Some(((*pos_r, *pos_c), best_buddy));
                            }
                        }
                    }
                    // Деталь справа
                    if *pos_c != 2 * img_width - 1 {
                        let (right_piece_r, right_piece_c) = tmp_chromosome[*pos_r][*pos_c + 1];
                        if right_piece_r != usize::MAX {
                            let best_buddy =
                                pieces_buddies[2][right_piece_r * img_width + right_piece_c];
                            let (pos_1_r, pos_1_c) =
                                pos_in_chromosome_1[right_piece_r][right_piece_c];
                            let (pos_2_r, pos_2_c) =
                                pos_in_chromosome_2[right_piece_r][right_piece_c];
                            if ((pos_1_c != 0 && chromosome_1[pos_1_r][pos_1_c - 1] == best_buddy)
                                || (pos_2_c != 0
                                    && chromosome_2[pos_2_r][pos_2_c - 1] == best_buddy))
                                && free_pieces.contains(&best_buddy)
                            {
                                return Some(((*pos_r, *pos_c), best_buddy));
                            }
                        }
                    }
                    // Деталь сверху
                    if *pos_r != 0 {
                        let (up_piece_r, up_piece_c) = tmp_chromosome[*pos_r - 1][*pos_c];
                        if up_piece_r != usize::MAX {
                            let best_buddy = pieces_buddies[1][up_piece_r * img_width + up_piece_c];
                            let (pos_1_r, pos_1_c) = pos_in_chromosome_1[up_piece_r][up_piece_c];
                            let (pos_2_r, pos_2_c) = pos_in_chromosome_2[up_piece_r][up_piece_c];
                            if ((pos_1_r != img_height - 1
                                && chromosome_1[pos_1_r + 1][pos_1_c] == best_buddy)
                                || (pos_2_r != img_height - 1
                                    && chromosome_2[pos_2_r + 1][pos_2_c] == best_buddy))
                                && free_pieces.contains(&best_buddy)
                            {
                                return Some(((*pos_r, *pos_c), best_buddy));
                            }
                        }
                    }
                    // Деталь снизу
                    if *pos_r != 2 * img_height - 1 {
                        let (down_piece_r, down_piece_c) = tmp_chromosome[*pos_r + 1][*pos_c];
                        if down_piece_r != usize::MAX {
                            let best_buddy =
                                pieces_buddies[3][down_piece_r * img_width + down_piece_c];
                            let (pos_1_r, pos_1_c) =
                                pos_in_chromosome_1[down_piece_r][down_piece_c];
                            let (pos_2_r, pos_2_c) =
                                pos_in_chromosome_2[down_piece_r][down_piece_c];
                            if ((pos_1_r != 0 && chromosome_1[pos_1_r - 1][pos_1_c] == best_buddy)
                                || (pos_2_r != 0
                                    && chromosome_2[pos_2_r - 1][pos_2_c] == best_buddy))
                                && free_pieces.contains(&best_buddy)
                            {
                                return Some(((*pos_r, *pos_c), best_buddy));
                            }
                        }
                    }
                    None
                })
                .choose(&mut rng)
            {
                selected_pos = Some(pos);
                selected_piece = Some(piece);
            }
        }

        // Phase 3 (most compatible)
        if selected_pos.is_none() {
            let (pos_r, pos_c) = *free_positions.iter().choose(&mut rng).unwrap();

            let mut best_piece = (usize::MAX, usize::MAX);
            let mut best_dissimilarity = f32::INFINITY;
            for (piece_r, piece_c) in &free_pieces {
                let mut res = 0.0f32;
                // Деталь слева
                if pos_c != 0 {
                    let (left_piece_r, left_piece_c) = tmp_chromosome[pos_r][pos_c - 1];
                    if left_piece_r != usize::MAX {
                        res += pieces_dissimilarity[0][left_piece_r * img_width + left_piece_c]
                            [piece_r * img_width + piece_c];
                    }
                }
                // Деталь справа
                if pos_c != 2 * img_width - 1 {
                    let (right_piece_r, right_piece_c) = tmp_chromosome[pos_r][pos_c + 1];
                    if right_piece_r != usize::MAX {
                        res += pieces_dissimilarity[0][piece_r * img_width + piece_c]
                            [right_piece_r * img_width + right_piece_c];
                    }
                }
                // Деталь сверху
                if pos_r != 0 {
                    let (up_piece_r, up_piece_c) = tmp_chromosome[pos_r - 1][pos_c];
                    if up_piece_r != usize::MAX {
                        res += pieces_dissimilarity[1][up_piece_r * img_width + up_piece_c]
                            [piece_r * img_width + piece_c];
                    }
                }
                // Деталь снизу
                if pos_r != 2 * img_height - 1 {
                    let (down_piece_r, down_piece_c) = tmp_chromosome[pos_r + 1][pos_c];
                    if down_piece_r != usize::MAX {
                        res += pieces_dissimilarity[1][piece_r * img_width + piece_c]
                            [down_piece_r * img_width + down_piece_c];
                    }
                }
                if res < best_dissimilarity {
                    best_piece = (*piece_r, *piece_c);
                    best_dissimilarity = res;
                }
            }
            selected_pos = Some((pos_r, pos_c));
            selected_piece = Some(best_piece);
        }

        let selected_pos = selected_pos.unwrap();
        let selected_piece = selected_piece.unwrap();
        let (selected_pos_r, selected_pos_c) = selected_pos;
        tmp_chromosome[selected_pos_r][selected_pos_c] = selected_piece;
        free_positions.retain(|x| *x != selected_pos);
        free_pieces.remove(&selected_piece);
        min_r = min(min_r, selected_pos_r);
        max_r = max(max_r, selected_pos_r);
        min_c = min(min_c, selected_pos_c);
        max_c = max(max_c, selected_pos_c);
        for dr in [-1isize, 1] {
            let new_r = ((selected_pos_r as isize) + dr) as usize;
            if 1 + max(max_r, new_r) - min(min_r, new_r) > img_height {
                continue;
            }
            if tmp_chromosome[new_r][selected_pos_c].0 == usize::MAX {
                free_positions.push((new_r, selected_pos_c));
            }
        }
        for dc in [-1isize, 1] {
            let new_c = ((selected_pos_c as isize) + dc) as usize;
            if 1 + max(max_c, new_c) - min(min_c, new_c) > img_width {
                continue;
            }
            if tmp_chromosome[selected_pos_r][new_c].0 == usize::MAX {
                free_positions.push((selected_pos_r, new_c));
            }
        }
    }

    assert!(free_pieces.is_empty());
    assert_eq!(1 + max_r - min_r, img_height);
    assert_eq!(1 + max_c - min_c, img_width);
    (min_r..=max_r)
        .map(|r| (min_c..=max_c).map(|c| tmp_chromosome[r][c]).collect())
        .collect()
}

pub fn algorithm_step(
    population_size: usize,
    rng: &mut Xoshiro256PlusPlus,
    img_width: usize,
    img_height: usize,
    image_generations_processed: usize,
    pieces_dissimilarity: &[Vec<Vec<f32>>; 2],
    pieces_buddies: &[Vec<(usize, usize)>; 4],
    current_generation: &[Chromosome],
) -> Vec<Chromosome> {
    // Создание нового поколения
    let mut new_generation: Vec<Chromosome> = if image_generations_processed == 0 {
        (0..population_size)
            .map(|_| generate_random_chromosome(img_width, img_height, rng))
            .collect()
    } else {
        // Отбор лучших хромосом
        let mut best_chromosomes: Vec<_> = current_generation
            .iter()
            .take(ELITISM_COUNT)
            .cloned()
            .collect();

        let indexes: Vec<usize> = (0..population_size).collect();
        // Оценки хромосом текущего поколения
        let curr_gen_dissimilarities: Vec<_> = current_generation
            .iter()
            .map(|chromosome| {
                chromosome_dissimilarity(img_width, img_height, pieces_dissimilarity, chromosome)
            })
            .collect();
        let max_dissimilarity = curr_gen_dissimilarities
            .iter()
            .cloned()
            .reduce(f32::max)
            .unwrap();

        // Выбранные для скрещивания предки
        let parents: Vec<_> = (0..(population_size - ELITISM_COUNT))
            .map(|_| {
                let mut iter = indexes
                    .choose_multiple_weighted(rng, 2, |i| {
                        max_dissimilarity - curr_gen_dissimilarities[*i]
                    })
                    .unwrap();
                (*iter.next().unwrap(), *iter.next().unwrap())
            })
            .collect();
        // Генераторы случайных чисел для каждого скрещивания
        let rngs: Vec<_> = (0..(population_size - ELITISM_COUNT))
            .map(|_| {
                let tmp = rng.clone();
                rng.jump();
                tmp
            })
            .collect();
        // Получение нового поколения скрещиванием
        let mut other_chromosomes: Vec<_> = parents
            .into_par_iter()
            .zip(rngs.into_par_iter())
            .map(|((i, j), rng)| {
                chromosomes_crossover(
                    img_width,
                    img_height,
                    pieces_dissimilarity,
                    pieces_buddies,
                    &current_generation[i],
                    &current_generation[j],
                    rng,
                )
            })
            .collect();

        // Объединение
        best_chromosomes.append(&mut other_chromosomes);
        best_chromosomes
    };
    // Сортировка хромосом по возрастанию оценки
    new_generation.sort_by_cached_key(|chromosome| {
        FloatOrd(chromosome_dissimilarity(
            img_width,
            img_height,
            pieces_dissimilarity,
            chromosome,
        ))
    });
    new_generation
}
