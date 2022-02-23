use std::cmp::{max, min};

use float_ord::FloatOrd;
use image::RgbaImage;
use indexmap::{IndexMap, IndexSet};
use rand::{prelude::IteratorRandom, seq::SliceRandom, Rng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::image_processing::get_lab_image;

pub type Chromosome = Vec<Vec<(usize, usize)>>;

const ELITISM_COUNT: usize = 4;
const MUTATION_RATE: f32 = 0.005;

// Вычисление совместимостей деталей
pub fn calculate_dissimilarities(
    image: &RgbaImage,
    img_width: usize,
    img_height: usize,
    piece_size: usize,
) -> [Vec<Vec<f32>>; 2] {
    let lab_pixels = get_lab_image(image);
    let image_width = img_width * piece_size;

    let two_pixels_dissimilarity =
        |i: usize, j: usize| lab_pixels[i].squared_distance(&lab_pixels[j]);

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
    pieces_dissimilarity: &[Vec<Vec<f32>>; 2],
) -> [Vec<(usize, usize)>; 4] {
    let get_buddies = |ind: usize| {
        pieces_dissimilarity[ind]
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.iter()
                    .enumerate()
                    .filter(|(j, _)| i != *j)
                    .reduce(|x, y| if x.1 <= y.1 { x } else { y })
                    .map(|(j, _)| (j / img_width, j % img_width))
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let right_buddies = get_buddies(0);
    let down_buddies = get_buddies(1);

    let get_opposite_buddies = |ind: usize| {
        (0..pieces_dissimilarity[ind].len())
            .map(|i| {
                pieces_dissimilarity[ind]
                    .iter()
                    .enumerate()
                    .map(|(j, v)| (j, v[i]))
                    .filter(|(j, _)| i != *j)
                    .reduce(|x, y| if x.1 <= y.1 { x } else { y })
                    .map(|(j, _)| (j / img_width, j % img_width))
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let left_buddies = get_opposite_buddies(0);
    let up_buddies = get_opposite_buddies(1);

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

    let mut free_pieces: IndexSet<_> = (0..img_height)
        .flat_map(|r| (0..img_width).map(move |c| (r, c)))
        .collect();
    let mut free_positions_phase_1 = IndexMap::new();
    let mut free_positions_phase_2 = IndexMap::new();
    let mut free_positions_unknown = IndexSet::new();
    let mut piece_to_pos_phase_1: IndexMap<(usize, usize), Vec<(usize, usize)>> = IndexMap::new();
    let mut piece_to_pos_phase_2: IndexMap<(usize, usize), Vec<(usize, usize)>> = IndexMap::new();

    let mut new_chromosome: Chromosome = (0..(2 * img_height))
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
    free_positions_unknown.insert((img_height, img_width));
    while !free_positions_phase_1.is_empty()
        || !free_positions_phase_2.is_empty()
        || !free_positions_unknown.is_empty()
    {
        let mut selected_pos = None;
        let mut selected_piece = None;

        // Первая деталь
        if free_positions_unknown.len() == 1
            && free_positions_unknown.contains(&(img_height, img_width))
        {
            selected_pos = Some((img_height, img_width));
            selected_piece = Some(start_piece);
        }

        // Phase 1 (both parents agree)
        if selected_pos.is_none() {
            let tmp: Vec<_> = free_positions_unknown
                .iter()
                .filter_map(|(pos_r, pos_c)| {
                    // Деталь слева
                    if *pos_c != 0 {
                        let (left_piece_r, left_piece_c) = new_chromosome[*pos_r][*pos_c - 1];
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
                        let (right_piece_r, right_piece_c) = new_chromosome[*pos_r][*pos_c + 1];
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
                        let (up_piece_r, up_piece_c) = new_chromosome[*pos_r - 1][*pos_c];
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
                        let (down_piece_r, down_piece_c) = new_chromosome[*pos_r + 1][*pos_c];
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
                .collect();
            for (pos, piece) in tmp {
                free_positions_unknown.remove(&pos);
                if let Some(v) = piece_to_pos_phase_1.get_mut(&piece) {
                    v.push(pos);
                } else {
                    piece_to_pos_phase_1.insert(piece, vec![pos]);
                }
                if let Some(old_piece) = free_positions_phase_1.insert(pos, piece) {
                    if let Some(v) = piece_to_pos_phase_1.get_mut(&old_piece) {
                        v.retain(|x| *x != pos);
                        if v.is_empty() {
                            piece_to_pos_phase_1.remove(&old_piece);
                        }
                    }
                }
            }

            if !free_positions_phase_1.is_empty() {
                let ind = rng.gen_range(0..free_positions_phase_1.len());
                let (pos, mut piece) = free_positions_phase_1.get_index(ind).unwrap();

                if rng.gen_range(0.0f32..1.0) <= MUTATION_RATE {
                    let ind = rng.gen_range(0..free_pieces.len());
                    piece = free_pieces.get_index(ind).unwrap();
                }

                assert!(free_pieces.contains(piece));
                selected_pos = Some(*pos);
                selected_piece = Some(*piece);
            }
        }

        // Phase 2 (best-buddy)
        if selected_pos.is_none() {
            let tmp: Vec<_> = free_positions_unknown
                .iter()
                .filter_map(|(pos_r, pos_c)| {
                    // Деталь слева
                    if *pos_c != 0 {
                        let (left_piece_r, left_piece_c) = new_chromosome[*pos_r][*pos_c - 1];
                        if left_piece_r != usize::MAX {
                            let best_buddy =
                                pieces_buddies[0][left_piece_r * img_width + left_piece_c];
                            let (pos_1_r, pos_1_c) =
                                pos_in_chromosome_1[left_piece_r][left_piece_c];
                            let (pos_2_r, pos_2_c) =
                                pos_in_chromosome_2[left_piece_r][left_piece_c];
                            if pieces_buddies[2][best_buddy.0 * img_width + best_buddy.1]
                                == (left_piece_r, left_piece_c)
                                && ((pos_1_c != img_width - 1
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
                        let (right_piece_r, right_piece_c) = new_chromosome[*pos_r][*pos_c + 1];
                        if right_piece_r != usize::MAX {
                            let best_buddy =
                                pieces_buddies[2][right_piece_r * img_width + right_piece_c];
                            let (pos_1_r, pos_1_c) =
                                pos_in_chromosome_1[right_piece_r][right_piece_c];
                            let (pos_2_r, pos_2_c) =
                                pos_in_chromosome_2[right_piece_r][right_piece_c];
                            if pieces_buddies[0][best_buddy.0 * img_width + best_buddy.1]
                                == (right_piece_r, right_piece_c)
                                && ((pos_1_c != 0
                                    && chromosome_1[pos_1_r][pos_1_c - 1] == best_buddy)
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
                        let (up_piece_r, up_piece_c) = new_chromosome[*pos_r - 1][*pos_c];
                        if up_piece_r != usize::MAX {
                            let best_buddy = pieces_buddies[1][up_piece_r * img_width + up_piece_c];
                            let (pos_1_r, pos_1_c) = pos_in_chromosome_1[up_piece_r][up_piece_c];
                            let (pos_2_r, pos_2_c) = pos_in_chromosome_2[up_piece_r][up_piece_c];
                            if pieces_buddies[3][best_buddy.0 * img_width + best_buddy.1]
                                == (up_piece_r, up_piece_c)
                                && ((pos_1_r != img_height - 1
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
                        let (down_piece_r, down_piece_c) = new_chromosome[*pos_r + 1][*pos_c];
                        if down_piece_r != usize::MAX {
                            let best_buddy =
                                pieces_buddies[3][down_piece_r * img_width + down_piece_c];
                            let (pos_1_r, pos_1_c) =
                                pos_in_chromosome_1[down_piece_r][down_piece_c];
                            let (pos_2_r, pos_2_c) =
                                pos_in_chromosome_2[down_piece_r][down_piece_c];
                            if pieces_buddies[1][best_buddy.0 * img_width + best_buddy.1]
                                == (down_piece_r, down_piece_c)
                                && ((pos_1_r != 0
                                    && chromosome_1[pos_1_r - 1][pos_1_c] == best_buddy)
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
                .collect();
            for (pos, piece) in tmp {
                free_positions_unknown.remove(&pos);
                if let Some(v) = piece_to_pos_phase_2.get_mut(&piece) {
                    v.push(pos);
                } else {
                    piece_to_pos_phase_2.insert(piece, vec![pos]);
                }
                if let Some(old_piece) = free_positions_phase_2.insert(pos, piece) {
                    if let Some(v) = piece_to_pos_phase_2.get_mut(&old_piece) {
                        v.retain(|x| *x != pos);
                        if v.is_empty() {
                            piece_to_pos_phase_2.remove(&old_piece);
                        }
                    }
                }
            }

            if !free_positions_phase_2.is_empty() {
                let ind = rng.gen_range(0..free_positions_phase_2.len());
                let (pos, piece) = free_positions_phase_2.get_index(ind).unwrap();

                assert!(free_pieces.contains(piece));
                selected_pos = Some(*pos);
                selected_piece = Some(*piece);
            }
        }

        // Phase 3 (most compatible)
        if selected_pos.is_none() {
            let ind = rng.gen_range(0..free_positions_unknown.len());
            let (pos_r, pos_c) = *free_positions_unknown.get_index(ind).unwrap();

            let mut best_piece = (usize::MAX, usize::MAX);
            let mut best_dissimilarity = f32::INFINITY;
            if rng.gen_range(0.0f32..1.0) <= MUTATION_RATE {
                let ind = rng.gen_range(0..free_pieces.len());
                best_piece = *free_pieces.get_index(ind).unwrap();
            } else {
                for (piece_r, piece_c) in &free_pieces {
                    let mut res = 0.0f32;
                    // Деталь слева
                    if pos_c != 0 {
                        let (left_piece_r, left_piece_c) = new_chromosome[pos_r][pos_c - 1];
                        if left_piece_r != usize::MAX {
                            res += pieces_dissimilarity[0][left_piece_r * img_width + left_piece_c]
                                [piece_r * img_width + piece_c];
                        }
                    }
                    // Деталь справа
                    if pos_c != 2 * img_width - 1 {
                        let (right_piece_r, right_piece_c) = new_chromosome[pos_r][pos_c + 1];
                        if right_piece_r != usize::MAX {
                            res += pieces_dissimilarity[0][piece_r * img_width + piece_c]
                                [right_piece_r * img_width + right_piece_c];
                        }
                    }
                    // Деталь сверху
                    if pos_r != 0 {
                        let (up_piece_r, up_piece_c) = new_chromosome[pos_r - 1][pos_c];
                        if up_piece_r != usize::MAX {
                            res += pieces_dissimilarity[1][up_piece_r * img_width + up_piece_c]
                                [piece_r * img_width + piece_c];
                        }
                    }
                    // Деталь снизу
                    if pos_r != 2 * img_height - 1 {
                        let (down_piece_r, down_piece_c) = new_chromosome[pos_r + 1][pos_c];
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
            }

            assert!(free_pieces.contains(&best_piece));
            selected_pos = Some((pos_r, pos_c));
            selected_piece = Some(best_piece);
        }

        let selected_pos = selected_pos.unwrap();
        let selected_piece = selected_piece.unwrap();
        let (selected_pos_r, selected_pos_c) = selected_pos;
        new_chromosome[selected_pos_r][selected_pos_c] = selected_piece;
        free_positions_phase_1.remove(&selected_pos);
        free_positions_phase_2.remove(&selected_pos);
        free_positions_unknown.remove(&selected_pos);
        free_pieces.remove(&selected_piece);
        if let Some(v) = piece_to_pos_phase_1.get(&selected_piece) {
            for pos in v {
                if free_positions_phase_1.contains_key(pos) {
                    free_positions_phase_1.remove(pos);
                    free_positions_unknown.insert(*pos);
                }
            }
            piece_to_pos_phase_1.remove(&selected_piece);
        }
        if let Some(v) = piece_to_pos_phase_2.get(&selected_piece) {
            for pos in v {
                if free_positions_phase_2.contains_key(pos) {
                    free_positions_phase_2.remove(pos);
                    free_positions_unknown.insert(*pos);
                }
            }
            piece_to_pos_phase_2.remove(&selected_piece);
        }

        if selected_pos_r < min_r {
            min_r = selected_pos_r;
            if 1 + max_r - min_r == img_height {
                for c in 0..(2 * img_width) {
                    free_positions_phase_1.remove(&(min_r - 1, c));
                    free_positions_phase_2.remove(&(min_r - 1, c));
                    free_positions_unknown.remove(&(min_r - 1, c));

                    free_positions_phase_1.remove(&(max_r + 1, c));
                    free_positions_phase_2.remove(&(max_r + 1, c));
                    free_positions_unknown.remove(&(max_r + 1, c));
                }
            }
        } else if selected_pos_r > max_r {
            max_r = selected_pos_r;
            if 1 + max_r - min_r == img_height {
                for c in 0..(2 * img_width) {
                    free_positions_phase_1.remove(&(min_r - 1, c));
                    free_positions_phase_2.remove(&(min_r - 1, c));
                    free_positions_unknown.remove(&(min_r - 1, c));

                    free_positions_phase_1.remove(&(max_r + 1, c));
                    free_positions_phase_2.remove(&(max_r + 1, c));
                    free_positions_unknown.remove(&(max_r + 1, c));
                }
            }
        }
        if selected_pos_c < min_c {
            min_c = selected_pos_c;
            if 1 + max_c - min_c == img_width {
                for r in 0..(2 * img_height) {
                    free_positions_phase_1.remove(&(r, min_c - 1));
                    free_positions_phase_2.remove(&(r, min_c - 1));
                    free_positions_unknown.remove(&(r, min_c - 1));

                    free_positions_phase_1.remove(&(r, max_c + 1));
                    free_positions_phase_2.remove(&(r, max_c + 1));
                    free_positions_unknown.remove(&(r, max_c + 1));
                }
            }
        } else if selected_pos_c > max_c {
            max_c = selected_pos_c;
            if 1 + max_c - min_c == img_width {
                for r in 0..(2 * img_height) {
                    free_positions_phase_1.remove(&(r, min_c - 1));
                    free_positions_phase_2.remove(&(r, min_c - 1));
                    free_positions_unknown.remove(&(r, min_c - 1));

                    free_positions_phase_1.remove(&(r, max_c + 1));
                    free_positions_phase_2.remove(&(r, max_c + 1));
                    free_positions_unknown.remove(&(r, max_c + 1));
                }
            }
        }

        for dr in [-1isize, 1] {
            let new_r = ((selected_pos_r as isize) + dr) as usize;
            if 1 + max(max_r, new_r) - min(min_r, new_r) > img_height {
                continue;
            }
            if new_chromosome[new_r][selected_pos_c].0 == usize::MAX
                && !free_positions_phase_1.contains_key(&(new_r, selected_pos_c))
                && !free_positions_phase_2.contains_key(&(new_r, selected_pos_c))
            {
                free_positions_unknown.insert((new_r, selected_pos_c));
            }
        }
        for dc in [-1isize, 1] {
            let new_c = ((selected_pos_c as isize) + dc) as usize;
            if 1 + max(max_c, new_c) - min(min_c, new_c) > img_width {
                continue;
            }
            if new_chromosome[selected_pos_r][new_c].0 == usize::MAX
                && !free_positions_phase_1.contains_key(&(selected_pos_r, new_c))
                && !free_positions_phase_2.contains_key(&(selected_pos_r, new_c))
            {
                free_positions_unknown.insert((selected_pos_r, new_c));
            }
        }
    }

    assert!(free_pieces.is_empty());
    assert_eq!(1 + max_r - min_r, img_height);
    assert_eq!(1 + max_c - min_c, img_width);
    (min_r..=max_r)
        .map(|r| (min_c..=max_c).map(|c| new_chromosome[r][c]).collect())
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
        let min_dissimilarity = curr_gen_dissimilarities
            .iter()
            .cloned()
            .reduce(f32::min)
            .unwrap();
        let max_dissimilarity = curr_gen_dissimilarities
            .iter()
            .cloned()
            .reduce(f32::max)
            .unwrap();
        println!("{}, {}", min_dissimilarity, max_dissimilarity);

        // Выбранные для скрещивания предки
        let parents: Vec<_> = (0..(population_size - ELITISM_COUNT))
            .map(|_| {
                let mut iter = indexes
                    .choose_multiple_weighted(rng, 2, |i| {
                        (-(curr_gen_dissimilarities[*i] - min_dissimilarity)
                            / (max_dissimilarity - min_dissimilarity))
                            * 0.9
                            + 0.95
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
