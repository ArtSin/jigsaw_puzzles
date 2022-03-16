use std::cmp::{max, min};

use float_ord::FloatOrd;
use fxhash::FxBuildHasher;
use image::RgbaImage;
use indexmap::{IndexMap, IndexSet};
use rand::{prelude::IteratorRandom, seq::SliceRandom, Rng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::image_processing::get_lab_image;

pub type Chromosome = Vec<(usize, usize)>;

const ELITISM_COUNT: usize = 4;
const MUTATION_RATE_1: f32 = 0.001;
const MUTATION_RATE_3: f32 = 0.005;

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

// Нахождение "лучших приятелей"
pub fn find_best_buddies(
    img_width: usize,
    pieces_dissimilarity: &[Vec<Vec<f32>>; 2],
) -> [Vec<(usize, usize)>; 4] {
    // Нахождение приятелей справа и снизу от детали
    let get_buddies = |ind: usize| {
        pieces_dissimilarity[ind]
            .par_iter()
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

    // Нахождение приятелей слева и сверху от детали
    let get_opposite_buddies = |ind: usize| {
        (0..pieces_dissimilarity[ind].len())
            .into_par_iter()
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
    // Несовместимости деталей по направлению вправо
    let right_dissimilarities = (0..img_height)
        .map(|i_r| {
            (0..(img_width - 1))
                .map(|i_c| {
                    let (j_r, j_c) = chromosome[i_r * img_width + i_c];
                    let (k_r, k_c) = chromosome[i_r * img_width + i_c + 1];
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
                    let (j_r, j_c) = chromosome[i_r * img_width + i_c];
                    let (k_r, k_c) = chromosome[(i_r + 1) * img_width + i_c];
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
    rand_nums
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
    // Случайная начальная деталь
    let start_piece = (
        (0..img_height).choose(&mut rng).unwrap(),
        (0..img_width).choose(&mut rng).unwrap(),
    );

    // Положение каждой детали в каждом предке
    let mut pos_in_chromosome_1 = vec![(usize::MAX, usize::MAX); img_width * img_height];
    let mut pos_in_chromosome_2 = pos_in_chromosome_1.clone();
    for r in 0..img_height {
        for c in 0..img_width {
            let (i, j) = chromosome_1[r * img_width + c];
            pos_in_chromosome_1[i * img_width + j] = (r, c);

            let (i, j) = chromosome_2[r * img_width + c];
            pos_in_chromosome_2[i * img_width + j] = (r, c);
        }
    }

    // Свободные детали
    let mut free_pieces: IndexSet<_, FxBuildHasher> = (0..img_height)
        .flat_map(|r| (0..img_width).map(move |c| (r, c)))
        .collect();
    // Свободные позиции, подходящие для фазы 1, и детали, которые можно туда поставить
    let mut free_positions_phase_1 = IndexMap::with_hasher(FxBuildHasher::default());
    // Свободные позиции, подходящие для фазы 2, и детали, которые можно туда поставить
    let mut free_positions_phase_2 = IndexMap::with_hasher(FxBuildHasher::default());
    // Свободные позиции, подходящие для фазы 3
    let mut free_positions_phase_3 = IndexSet::with_hasher(FxBuildHasher::default());
    // Свободные позиции, не подходящие для фазы 1, которые должны быть рассмотрены в фазе 2 или 3
    let mut free_positions_not_in_phase_1 = IndexSet::with_hasher(FxBuildHasher::default());
    // Свободные нерассмотренные позиции
    let mut free_positions_unknown: IndexSet<(usize, usize), _> =
        IndexSet::with_hasher(FxBuildHasher::default());
    // Позиции, не подходящие для фазы 1
    let mut bad_positions_phase_1 = vec![false; 2 * img_width * 2 * img_height];
    // Позиции, не подходящие для фазы 2
    let mut bad_positions_phase_2 = vec![false; 2 * img_width * 2 * img_height];
    // Соответствие деталям позиций, подходящих для фазы 1
    let mut piece_to_pos_phase_1: IndexMap<(usize, usize), Vec<(usize, usize)>, _> =
        IndexMap::with_hasher(FxBuildHasher::default());
    // Соответствие деталям позиций, подходящих для фазы 2
    let mut piece_to_pos_phase_2: IndexMap<(usize, usize), Vec<(usize, usize)>, _> =
        IndexMap::with_hasher(FxBuildHasher::default());

    // Новая хромосома, получаемая в результате скрещивания
    // Так как неизвестно положение начальной детали в исходном изображении,
    // и построение может идти в любую сторону, высота и ширина в 2 раза больше необходимой
    let mut new_chromosome: Chromosome =
        vec![(usize::MAX, usize::MAX); 2 * img_width * 2 * img_height];

    // Текущие грани построенного изображения
    let (mut min_r, mut max_r, mut min_c, mut max_c) =
        (img_height, img_height, img_width, img_width);
    // Флаг, обозначающий необходимость добавления начальной детали в центр
    let mut start_flag = true;
    // Пока есть свободные позиции
    while start_flag
        || !free_positions_phase_1.is_empty()
        || !free_positions_phase_2.is_empty()
        || !free_positions_phase_3.is_empty()
        || !free_positions_not_in_phase_1.is_empty()
        || !free_positions_unknown.is_empty()
    {
        // Выбираемая на данной итерации позиция и деталь
        let mut selected_pos = None;
        let mut selected_piece = None;

        // Начальная деталь
        if start_flag {
            start_flag = false;
            selected_pos = Some((img_height, img_width));
            selected_piece = Some(start_piece);
        }

        // Фаза 1 (у обоих предков одинаковая деталь в определённом направлении от позиции)
        if selected_pos.is_none() {
            // Обработка нерассмотренных свободных позиций
            let tmp = free_positions_unknown.iter().map(|(pos_r, pos_c)| {
                let (pos_r, pos_c) = (*pos_r, *pos_c);
                assert!(new_chromosome[pos_r * 2 * img_width + pos_c].0 == usize::MAX);

                // Обработка детали слева
                if pos_c != 0 {
                    // Деталь слева в новой хромосоме
                    let (left_piece_r, left_piece_c) =
                        new_chromosome[pos_r * 2 * img_width + pos_c - 1];
                    if left_piece_r != usize::MAX {
                        // Положение этой детали в предках
                        let (pos_1_r, pos_1_c) =
                            pos_in_chromosome_1[left_piece_r * img_width + left_piece_c];
                        let (pos_2_r, pos_2_c) =
                            pos_in_chromosome_2[left_piece_r * img_width + left_piece_c];
                        // Если справа от левой детали в обоих предках одна и та же свободная деталь, то выбираем её
                        if pos_1_c != img_width - 1
                            && pos_2_c != img_width - 1
                            && chromosome_1[pos_1_r * img_width + pos_1_c + 1]
                                == chromosome_2[pos_2_r * img_width + pos_2_c + 1]
                            && free_pieces
                                .contains(&chromosome_1[pos_1_r * img_width + pos_1_c + 1])
                        {
                            return Ok((
                                (pos_r, pos_c),
                                chromosome_1[pos_1_r * img_width + pos_1_c + 1],
                            ));
                        }
                    }
                }
                // Обработка детали справа
                if pos_c != 2 * img_width - 1 {
                    // Деталь справа в новой хромосоме
                    let (right_piece_r, right_piece_c) =
                        new_chromosome[pos_r * 2 * img_width + pos_c + 1];
                    if right_piece_r != usize::MAX {
                        // Положение этой детали в предках
                        let (pos_1_r, pos_1_c) =
                            pos_in_chromosome_1[right_piece_r * img_width + right_piece_c];
                        let (pos_2_r, pos_2_c) =
                            pos_in_chromosome_2[right_piece_r * img_width + right_piece_c];
                        // Если слева от правой детали в обоих предках одна и та же свободная деталь, то выбираем её
                        if pos_1_c != 0
                            && pos_2_c != 0
                            && chromosome_1[pos_1_r * img_width + pos_1_c - 1]
                                == chromosome_2[pos_2_r * img_width + pos_2_c - 1]
                            && free_pieces
                                .contains(&chromosome_1[pos_1_r * img_width + pos_1_c - 1])
                        {
                            return Ok((
                                (pos_r, pos_c),
                                chromosome_1[pos_1_r * img_width + pos_1_c - 1],
                            ));
                        }
                    }
                }
                // Обработка детали сверху
                if pos_r != 0 {
                    // Деталь сверху в новой хромосоме
                    let (up_piece_r, up_piece_c) =
                        new_chromosome[(pos_r - 1) * 2 * img_width + pos_c];
                    if up_piece_r != usize::MAX {
                        // Положение этой детали в предках
                        let (pos_1_r, pos_1_c) =
                            pos_in_chromosome_1[up_piece_r * img_width + up_piece_c];
                        let (pos_2_r, pos_2_c) =
                            pos_in_chromosome_2[up_piece_r * img_width + up_piece_c];
                        // Если снизу от верхней детали в обоих предках одна и та же свободная деталь, то выбираем её
                        if pos_1_r != img_height - 1
                            && pos_2_r != img_height - 1
                            && chromosome_1[(pos_1_r + 1) * img_width + pos_1_c]
                                == chromosome_2[(pos_2_r + 1) * img_width + pos_2_c]
                            && free_pieces
                                .contains(&chromosome_1[(pos_1_r + 1) * img_width + pos_1_c])
                        {
                            return Ok((
                                (pos_r, pos_c),
                                chromosome_1[(pos_1_r + 1) * img_width + pos_1_c],
                            ));
                        }
                    }
                }
                // Обработка детали снизу
                if pos_r != 2 * img_height - 1 {
                    // Деталь снизу в новой хромосоме
                    let (down_piece_r, down_piece_c) =
                        new_chromosome[(pos_r + 1) * 2 * img_width + pos_c];
                    if down_piece_r != usize::MAX {
                        // Положение этой детали в предках
                        let (pos_1_r, pos_1_c) =
                            pos_in_chromosome_1[down_piece_r * img_width + down_piece_c];
                        let (pos_2_r, pos_2_c) =
                            pos_in_chromosome_2[down_piece_r * img_width + down_piece_c];
                        // Если сверху от нижней детали в обоих предках одна и та же свободная деталь, то выбираем её
                        if pos_1_r != 0
                            && pos_2_r != 0
                            && chromosome_1[(pos_1_r - 1) * img_width + pos_1_c]
                                == chromosome_2[(pos_2_r - 1) * img_width + pos_2_c]
                            && free_pieces
                                .contains(&chromosome_1[(pos_1_r - 1) * img_width + pos_1_c])
                        {
                            return Ok((
                                (pos_r, pos_c),
                                chromosome_1[(pos_1_r - 1) * img_width + pos_1_c],
                            ));
                        }
                    }
                }
                // Позиция не подходит для фазы 1
                Err((pos_r, pos_c))
            });

            for res in tmp {
                match res {
                    // Если позиция подходит для фазы 1, то добавление её
                    // в список позиций фазы 1 и соответствие деталей позициям фазы 1
                    Ok((pos, piece)) => {
                        if let Some(v) = piece_to_pos_phase_1.get_mut(&piece) {
                            v.push(pos);
                        } else {
                            piece_to_pos_phase_1.insert(piece, vec![pos]);
                        }
                        assert!(free_positions_phase_1.insert(pos, piece).is_none());
                    }
                    // Если не подходит, то добавление позиции в список не подходящих для фазы 1
                    // и тех, которые должны быть рассмотрены в фазе 2 или 3
                    Err(pos) => {
                        bad_positions_phase_1[pos.0 * 2 * img_width + pos.1] = true;
                        free_positions_not_in_phase_1.insert(pos);
                    }
                }
            }
            // Нерассмотренных позиций нет
            free_positions_unknown.clear();

            // Если есть хотя бы одна подходящая позиция
            if !free_positions_phase_1.is_empty() {
                // Выбор случайной позиции
                let ind = rng.gen_range(0..free_positions_phase_1.len());
                let (pos, mut piece) = free_positions_phase_1.get_index(ind).unwrap();

                // С небольшой вероятностью происходит мутация: выбирается случайная деталь
                if rng.gen_range(0.0f32..1.0) <= MUTATION_RATE_1 {
                    let ind = rng.gen_range(0..free_pieces.len());
                    piece = free_pieces.get_index(ind).unwrap();
                }

                assert!(new_chromosome[pos.0 * 2 * img_width + pos.1].0 == usize::MAX);
                assert!(free_pieces.contains(piece));
                // Позиция и деталь на данной итерации выбраны
                selected_pos = Some(*pos);
                selected_piece = Some(*piece);
            }
        }

        // Фаза 2 (две детали являются "лучшими приятелями" друг друга)
        if selected_pos.is_none() {
            // Обработка свободных позиций, не подходящих для фазы 1
            let tmp = free_positions_not_in_phase_1.iter().map(|(pos_r, pos_c)| {
                let (pos_r, pos_c) = (*pos_r, *pos_c);
                assert!(new_chromosome[pos_r * 2 * img_width + pos_c].0 == usize::MAX);

                // Обработка детали слева
                if pos_c != 0 {
                    // Деталь слева в новой хромосоме
                    let (left_piece_r, left_piece_c) =
                        new_chromosome[pos_r * 2 * img_width + pos_c - 1];
                    if left_piece_r != usize::MAX {
                        // "Лучший приятель" этой детали
                        let best_buddy = pieces_buddies[0][left_piece_r * img_width + left_piece_c];
                        // Положение левой детали в предках
                        let (pos_1_r, pos_1_c) =
                            pos_in_chromosome_1[left_piece_r * img_width + left_piece_c];
                        let (pos_2_r, pos_2_c) =
                            pos_in_chromosome_2[left_piece_r * img_width + left_piece_c];
                        // Если найденный "лучший приятель" также считает левую деталь "лучшим приятелем",
                        // и справа от левой детали хотя бы в одном из предков есть этот "лучший приятель",
                        // и он свободен, то выбираем его
                        if pieces_buddies[2][best_buddy.0 * img_width + best_buddy.1]
                            == (left_piece_r, left_piece_c)
                            && ((pos_1_c != img_width - 1
                                && chromosome_1[pos_1_r * img_width + pos_1_c + 1] == best_buddy)
                                || (pos_2_c != img_width - 1
                                    && chromosome_2[pos_2_r * img_width + pos_2_c + 1]
                                        == best_buddy))
                            && free_pieces.contains(&best_buddy)
                        {
                            return Ok(((pos_r, pos_c), best_buddy));
                        }
                    }
                }
                // Обработка детали справа
                if pos_c != 2 * img_width - 1 {
                    // Деталь справа в новой хромосоме
                    let (right_piece_r, right_piece_c) =
                        new_chromosome[pos_r * 2 * img_width + pos_c + 1];
                    if right_piece_r != usize::MAX {
                        // "Лучший приятель" этой детали
                        let best_buddy =
                            pieces_buddies[2][right_piece_r * img_width + right_piece_c];
                        // Положение правой детали в предках
                        let (pos_1_r, pos_1_c) =
                            pos_in_chromosome_1[right_piece_r * img_width + right_piece_c];
                        let (pos_2_r, pos_2_c) =
                            pos_in_chromosome_2[right_piece_r * img_width + right_piece_c];
                        // Если найденный "лучший приятель" также считает правую деталь "лучшим приятелем",
                        // и слева от правой детали хотя бы в одном из предков есть этот "лучший приятель",
                        // и он свободен, то выбираем его
                        if pieces_buddies[0][best_buddy.0 * img_width + best_buddy.1]
                            == (right_piece_r, right_piece_c)
                            && ((pos_1_c != 0
                                && chromosome_1[pos_1_r * img_width + pos_1_c - 1] == best_buddy)
                                || (pos_2_c != 0
                                    && chromosome_2[pos_2_r * img_width + pos_2_c - 1]
                                        == best_buddy))
                            && free_pieces.contains(&best_buddy)
                        {
                            return Ok(((pos_r, pos_c), best_buddy));
                        }
                    }
                }
                // Обработка детали сверху
                if pos_r != 0 {
                    // Деталь сверху в новой хромосоме
                    let (up_piece_r, up_piece_c) =
                        new_chromosome[(pos_r - 1) * 2 * img_width + pos_c];
                    if up_piece_r != usize::MAX {
                        // "Лучший приятель" этой детали
                        let best_buddy = pieces_buddies[1][up_piece_r * img_width + up_piece_c];
                        // Положение верхней детали в предках
                        let (pos_1_r, pos_1_c) =
                            pos_in_chromosome_1[up_piece_r * img_width + up_piece_c];
                        let (pos_2_r, pos_2_c) =
                            pos_in_chromosome_2[up_piece_r * img_width + up_piece_c];
                        // Если найденный "лучший приятель" также считает верхнюю деталь "лучшим приятелем",
                        // и снизу от верхней детали хотя бы в одном из предков есть этот "лучший приятель",
                        // и он свободен, то выбираем его
                        if pieces_buddies[3][best_buddy.0 * img_width + best_buddy.1]
                            == (up_piece_r, up_piece_c)
                            && ((pos_1_r != img_height - 1
                                && chromosome_1[(pos_1_r + 1) * img_width + pos_1_c] == best_buddy)
                                || (pos_2_r != img_height - 1
                                    && chromosome_2[(pos_2_r + 1) * img_width + pos_2_c]
                                        == best_buddy))
                            && free_pieces.contains(&best_buddy)
                        {
                            return Ok(((pos_r, pos_c), best_buddy));
                        }
                    }
                }
                // Обработка детали снизу
                if pos_r != 2 * img_height - 1 {
                    // Деталь снизу в новой хромосоме
                    let (down_piece_r, down_piece_c) =
                        new_chromosome[(pos_r + 1) * 2 * img_width + pos_c];
                    if down_piece_r != usize::MAX {
                        // "Лучший приятель" этой детали
                        let best_buddy = pieces_buddies[3][down_piece_r * img_width + down_piece_c];
                        // Положение нижней детали в предках
                        let (pos_1_r, pos_1_c) =
                            pos_in_chromosome_1[down_piece_r * img_width + down_piece_c];
                        let (pos_2_r, pos_2_c) =
                            pos_in_chromosome_2[down_piece_r * img_width + down_piece_c];
                        // Если найденный "лучший приятель" также считает нижнюю деталь "лучшим приятелем",
                        // и сверху от нижней детали хотя бы в одном из предков есть этот "лучший приятель",
                        // и он свободен, то выбираем его
                        if pieces_buddies[1][best_buddy.0 * img_width + best_buddy.1]
                            == (down_piece_r, down_piece_c)
                            && ((pos_1_r != 0
                                && chromosome_1[(pos_1_r - 1) * img_width + pos_1_c] == best_buddy)
                                || (pos_2_r != 0
                                    && chromosome_2[(pos_2_r - 1) * img_width + pos_2_c]
                                        == best_buddy))
                            && free_pieces.contains(&best_buddy)
                        {
                            return Ok(((pos_r, pos_c), best_buddy));
                        }
                    }
                }
                // Позиция не подходит для фазы 2
                Err((pos_r, pos_c))
            });

            for res in tmp {
                match res {
                    // Если позиция подходит для фазы 2, то добавление её
                    // в список позиций фазы 2 и соответствие деталей позициям фазы 2
                    Ok((pos, piece)) => {
                        if let Some(v) = piece_to_pos_phase_2.get_mut(&piece) {
                            v.push(pos);
                        } else {
                            piece_to_pos_phase_2.insert(piece, vec![pos]);
                        }
                        assert!(free_positions_phase_2.insert(pos, piece).is_none());
                    }
                    // Если не подходит, то добавление позиции в список не подходящих для фазы 2
                    // и подходящих для фазы 3
                    Err(pos) => {
                        bad_positions_phase_2[pos.0 * 2 * img_width + pos.1] = true;
                        free_positions_phase_3.insert(pos);
                    }
                }
            }
            // Нерассмотренных позиций нет
            free_positions_not_in_phase_1.clear();

            // Если есть хотя бы одна подходящая позиция
            if !free_positions_phase_2.is_empty() {
                // Выбор случайной позиции
                let ind = rng.gen_range(0..free_positions_phase_2.len());
                let (pos, piece) = free_positions_phase_2.get_index(ind).unwrap();

                assert!(new_chromosome[pos.0 * 2 * img_width + pos.1].0 == usize::MAX);
                assert!(free_pieces.contains(piece));
                // Позиция и деталь на данной итерации выбраны
                selected_pos = Some(*pos);
                selected_piece = Some(*piece);
            }
        }

        // Фаза 3 (наиболее подходящая деталь для позиции)
        if selected_pos.is_none() {
            assert!(!free_positions_phase_3.is_empty());
            assert!(!free_pieces.is_empty());

            // Выбор случайной позиции
            let ind = rng.gen_range(0..free_positions_phase_3.len());
            let (pos_r, pos_c) = *free_positions_phase_3.get_index(ind).unwrap();

            // Наиболее подходящая деталь
            let mut best_piece = (usize::MAX, usize::MAX);
            // С небольшой вероятностью происходит мутация: выбирается случайная деталь
            if rng.gen_range(0.0f32..1.0) <= MUTATION_RATE_3 {
                let ind = rng.gen_range(0..free_pieces.len());
                best_piece = *free_pieces.get_index(ind).unwrap();
            } else {
                // Деталь слева в новой хромосоме
                let (left_piece_r, left_piece_c) = if pos_c == 0 {
                    (usize::MAX, usize::MAX)
                } else {
                    new_chromosome[pos_r * 2 * img_width + pos_c - 1]
                };
                // Деталь справа в новой хромосоме
                let (right_piece_r, right_piece_c) = if pos_c == 2 * img_width - 1 {
                    (usize::MAX, usize::MAX)
                } else {
                    new_chromosome[pos_r * 2 * img_width + pos_c + 1]
                };
                // Деталь сверху в новой хромосоме
                let (up_piece_r, up_piece_c) = if pos_r == 0 {
                    (usize::MAX, usize::MAX)
                } else {
                    new_chromosome[(pos_r - 1) * 2 * img_width + pos_c]
                };
                // Деталь снизу в новой хромосоме
                let (down_piece_r, down_piece_c) = if pos_r == 2 * img_height - 1 {
                    (usize::MAX, usize::MAX)
                } else {
                    new_chromosome[(pos_r + 1) * 2 * img_width + pos_c]
                };

                // Наименьшая несовместимость
                let mut best_dissimilarity = f32::INFINITY;
                // "Лучшие приятели"
                let mut buddies = Vec::new();
                if left_piece_r != usize::MAX {
                    buddies.push(pieces_buddies[0][left_piece_r * img_width + left_piece_c]);
                }
                if right_piece_r != usize::MAX {
                    buddies.push(pieces_buddies[2][right_piece_r * img_width + right_piece_c]);
                }
                if up_piece_r != usize::MAX {
                    buddies.push(pieces_buddies[1][up_piece_r * img_width + up_piece_c]);
                }
                if down_piece_r != usize::MAX {
                    buddies.push(pieces_buddies[3][down_piece_r * img_width + down_piece_c]);
                }
                let buddies = buddies.iter().filter(|piece| free_pieces.contains(*piece));

                // Обработка всех свободных деталей
                for (piece_r, piece_c) in buddies.chain(free_pieces.iter()) {
                    // Несовместимость
                    let mut res = 0.0f32;
                    // Обработка детали слева
                    if left_piece_r != usize::MAX {
                        res += pieces_dissimilarity[0][left_piece_r * img_width + left_piece_c]
                            [piece_r * img_width + piece_c];
                        if res >= best_dissimilarity {
                            continue;
                        }
                    }
                    // Обработка детали справа
                    if right_piece_r != usize::MAX {
                        res += pieces_dissimilarity[0][piece_r * img_width + piece_c]
                            [right_piece_r * img_width + right_piece_c];
                        if res >= best_dissimilarity {
                            continue;
                        }
                    }
                    // Обработка детали сверху
                    if up_piece_r != usize::MAX {
                        res += pieces_dissimilarity[1][up_piece_r * img_width + up_piece_c]
                            [piece_r * img_width + piece_c];
                        if res >= best_dissimilarity {
                            continue;
                        }
                    }
                    // Обработка детали снизу
                    if down_piece_r != usize::MAX {
                        res += pieces_dissimilarity[1][piece_r * img_width + piece_c]
                            [down_piece_r * img_width + down_piece_c];
                        if res >= best_dissimilarity {
                            continue;
                        }
                    }

                    // Обновление наименее несовместимой детали
                    best_piece = (*piece_r, *piece_c);
                    best_dissimilarity = res;
                }
            }

            assert!(new_chromosome[pos_r * 2 * img_width + pos_c].0 == usize::MAX);
            assert!(free_pieces.contains(&best_piece));
            // Позиция и деталь на данной итерации выбраны
            selected_pos = Some((pos_r, pos_c));
            selected_piece = Some(best_piece);
        }

        // Выбранные позиция и деталь
        let selected_pos = selected_pos.unwrap();
        let selected_piece = selected_piece.unwrap();
        let (selected_pos_r, selected_pos_c) = selected_pos;
        // Установка детали в новой хромосоме
        new_chromosome[selected_pos_r * 2 * img_width + selected_pos_c] = selected_piece;
        // Удаление позиции из списков свободных позиций
        free_positions_phase_1.remove(&selected_pos);
        free_positions_phase_2.remove(&selected_pos);
        free_positions_phase_3.remove(&selected_pos);
        free_positions_unknown.remove(&selected_pos);
        // Удаление детали из списка свободных
        free_pieces.remove(&selected_piece);
        // Если выбранной детали соответствуют позиции в фазе 1, то нужно рассмотреть их заново
        if let Some(v) = piece_to_pos_phase_1.get(&selected_piece) {
            for pos in v {
                if free_positions_phase_1.contains_key(pos) {
                    free_positions_phase_1.remove(pos);
                    free_positions_phase_3.remove(pos);
                    free_positions_not_in_phase_1.remove(pos);
                    bad_positions_phase_1[pos.0 * 2 * img_width + pos.1] = false;
                    bad_positions_phase_2[pos.0 * 2 * img_width + pos.1] = false;
                    if new_chromosome[pos.0 * 2 * img_width + pos.1].0 == usize::MAX {
                        free_positions_unknown.insert(*pos);
                    }
                }
            }
            piece_to_pos_phase_1.remove(&selected_piece);
        }
        // Если выбранной детали соответствуют позиции в фазе 2, то нужно рассмотреть их заново
        if let Some(v) = piece_to_pos_phase_2.get(&selected_piece) {
            for pos in v {
                if free_positions_phase_2.contains_key(pos) {
                    free_positions_phase_2.remove(pos);
                    free_positions_phase_3.remove(pos);
                    free_positions_not_in_phase_1.remove(pos);
                    bad_positions_phase_1[pos.0 * 2 * img_width + pos.1] = false;
                    bad_positions_phase_2[pos.0 * 2 * img_width + pos.1] = false;
                    if new_chromosome[pos.0 * 2 * img_width + pos.1].0 == usize::MAX {
                        free_positions_unknown.insert(*pos);
                    }
                }
            }
            piece_to_pos_phase_2.remove(&selected_piece);
        }

        // Обновление верхней границы
        if selected_pos_r < min_r {
            min_r = selected_pos_r;
            // Если достигнута полная высота изображения, удалить все позиции, выходящие за границы
            if 1 + max_r - min_r == img_height {
                for c in 0..(2 * img_width) {
                    if min_r >= 1 {
                        free_positions_phase_1.remove(&(min_r - 1, c));
                        free_positions_phase_2.remove(&(min_r - 1, c));
                        free_positions_phase_3.remove(&(min_r - 1, c));
                        free_positions_not_in_phase_1.remove(&(min_r - 1, c));
                        free_positions_unknown.remove(&(min_r - 1, c));
                        bad_positions_phase_1[(min_r - 1) * 2 * img_width + c] = false;
                        bad_positions_phase_2[(min_r - 1) * 2 * img_width + c] = false;
                    }

                    if max_r + 1 < 2 * img_height {
                        free_positions_phase_1.remove(&(max_r + 1, c));
                        free_positions_phase_2.remove(&(max_r + 1, c));
                        free_positions_phase_3.remove(&(max_r + 1, c));
                        free_positions_not_in_phase_1.remove(&(max_r + 1, c));
                        free_positions_unknown.remove(&(max_r + 1, c));
                        bad_positions_phase_1[(max_r + 1) * 2 * img_width + c] = false;
                        bad_positions_phase_2[(max_r + 1) * 2 * img_width + c] = false;
                    }
                }
            }
        }
        // Обновление нижней границы
        else if selected_pos_r > max_r {
            max_r = selected_pos_r;
            // Если достигнута полная высота изображения, удалить все позиции, выходящие за границы
            if 1 + max_r - min_r == img_height {
                for c in 0..(2 * img_width) {
                    if min_r >= 1 {
                        free_positions_phase_1.remove(&(min_r - 1, c));
                        free_positions_phase_2.remove(&(min_r - 1, c));
                        free_positions_phase_3.remove(&(min_r - 1, c));
                        free_positions_not_in_phase_1.remove(&(min_r - 1, c));
                        free_positions_unknown.remove(&(min_r - 1, c));
                        bad_positions_phase_1[(min_r - 1) * 2 * img_width + c] = false;
                        bad_positions_phase_2[(min_r - 1) * 2 * img_width + c] = false;
                    }

                    if max_r + 1 < 2 * img_height {
                        free_positions_phase_1.remove(&(max_r + 1, c));
                        free_positions_phase_2.remove(&(max_r + 1, c));
                        free_positions_phase_3.remove(&(max_r + 1, c));
                        free_positions_not_in_phase_1.remove(&(max_r + 1, c));
                        free_positions_unknown.remove(&(max_r + 1, c));
                        bad_positions_phase_1[(max_r + 1) * 2 * img_width + c] = false;
                        bad_positions_phase_2[(max_r + 1) * 2 * img_width + c] = false;
                    }
                }
            }
        }
        // Обновление левой границы
        if selected_pos_c < min_c {
            min_c = selected_pos_c;
            // Если достигнута полная ширина изображения, удалить все позиции, выходящие за границы
            if 1 + max_c - min_c == img_width {
                for r in 0..(2 * img_height) {
                    if min_c >= 1 {
                        free_positions_phase_1.remove(&(r, min_c - 1));
                        free_positions_phase_2.remove(&(r, min_c - 1));
                        free_positions_phase_3.remove(&(r, min_c - 1));
                        free_positions_not_in_phase_1.remove(&(r, min_c - 1));
                        free_positions_unknown.remove(&(r, min_c - 1));
                        bad_positions_phase_1[r * 2 * img_width + min_c - 1] = false;
                        bad_positions_phase_2[r * 2 * img_width + min_c - 1] = false;
                    }

                    if max_c + 1 < 2 * img_width {
                        free_positions_phase_1.remove(&(r, max_c + 1));
                        free_positions_phase_2.remove(&(r, max_c + 1));
                        free_positions_phase_3.remove(&(r, max_c + 1));
                        free_positions_not_in_phase_1.remove(&(r, max_c + 1));
                        free_positions_unknown.remove(&(r, max_c + 1));
                        bad_positions_phase_1[r * 2 * img_width + max_c + 1] = false;
                        bad_positions_phase_2[r * 2 * img_width + max_c + 1] = false;
                    }
                }
            }
        }
        // Обновление правой границы
        else if selected_pos_c > max_c {
            max_c = selected_pos_c;
            // Если достигнута полная ширина изображения, удалить все позиции, выходящие за границы
            if 1 + max_c - min_c == img_width {
                for r in 0..(2 * img_height) {
                    if min_c >= 1 {
                        free_positions_phase_1.remove(&(r, min_c - 1));
                        free_positions_phase_2.remove(&(r, min_c - 1));
                        free_positions_phase_3.remove(&(r, min_c - 1));
                        free_positions_not_in_phase_1.remove(&(r, min_c - 1));
                        free_positions_unknown.remove(&(r, min_c - 1));
                        bad_positions_phase_1[r * 2 * img_width + min_c - 1] = false;
                        bad_positions_phase_2[r * 2 * img_width + min_c - 1] = false;
                    }

                    if max_c + 1 < 2 * img_width {
                        free_positions_phase_1.remove(&(r, max_c + 1));
                        free_positions_phase_2.remove(&(r, max_c + 1));
                        free_positions_phase_3.remove(&(r, max_c + 1));
                        free_positions_not_in_phase_1.remove(&(r, max_c + 1));
                        free_positions_unknown.remove(&(r, max_c + 1));
                        bad_positions_phase_1[r * 2 * img_width + max_c + 1] = false;
                        bad_positions_phase_2[r * 2 * img_width + max_c + 1] = false;
                    }
                }
            }
        }

        // Обработка позиций сверху и снизу от выбранной
        for dr in [-1isize, 1] {
            let new_r = ((selected_pos_r as isize) + dr) as usize;
            if 1 + max(max_r, new_r) - min(min_r, new_r) > img_height {
                continue;
            }
            // Если позиция свободна и не подходит для фазы 1 или 2 (нет в списках подходящих для этих фаз
            // или уже была рассмотрена и оказалась неподходящей), то добавить её к рассмотрению,
            // удалив из всех списков рассмотренных позиций
            if new_chromosome[new_r * 2 * img_width + selected_pos_c].0 == usize::MAX
                && ((!free_positions_phase_1.contains_key(&(new_r, selected_pos_c))
                    && !free_positions_phase_2.contains_key(&(new_r, selected_pos_c)))
                    || bad_positions_phase_1[new_r * 2 * img_width + selected_pos_c]
                    || bad_positions_phase_2[new_r * 2 * img_width + selected_pos_c])
            {
                free_positions_phase_1.remove(&(new_r, selected_pos_c));
                free_positions_phase_2.remove(&(new_r, selected_pos_c));
                free_positions_phase_3.remove(&(new_r, selected_pos_c));
                free_positions_not_in_phase_1.remove(&(new_r, selected_pos_c));
                bad_positions_phase_1[new_r * 2 * img_width + selected_pos_c] = false;
                bad_positions_phase_2[new_r * 2 * img_width + selected_pos_c] = false;
                free_positions_unknown.insert((new_r, selected_pos_c));
            }
        }
        // Обработка позиций слева и справа от выбранной
        for dc in [-1isize, 1] {
            let new_c = ((selected_pos_c as isize) + dc) as usize;
            if 1 + max(max_c, new_c) - min(min_c, new_c) > img_width {
                continue;
            }
            // Если позиция свободна и не подходит для фазы 1 или 2 (нет в списках подходящих для этих фаз
            // или уже была рассмотрена и оказалась неподходящей), то добавить её к рассмотрению,
            // удалив из всех списков рассмотренных позиций
            if new_chromosome[selected_pos_r * 2 * img_width + new_c].0 == usize::MAX
                && ((!free_positions_phase_1.contains_key(&(selected_pos_r, new_c))
                    && !free_positions_phase_2.contains_key(&(selected_pos_r, new_c)))
                    || bad_positions_phase_1[selected_pos_r * 2 * img_width + new_c]
                    || bad_positions_phase_2[selected_pos_r * 2 * img_width + new_c])
            {
                free_positions_phase_1.remove(&(selected_pos_r, new_c));
                free_positions_phase_2.remove(&(selected_pos_r, new_c));
                free_positions_phase_3.remove(&(selected_pos_r, new_c));
                free_positions_not_in_phase_1.remove(&(selected_pos_r, new_c));
                bad_positions_phase_1[selected_pos_r * 2 * img_width + new_c] = false;
                bad_positions_phase_2[selected_pos_r * 2 * img_width + new_c] = false;
                free_positions_unknown.insert((selected_pos_r, new_c));
            }
        }
    }

    assert!(free_pieces.is_empty());
    assert_eq!(1 + max_r - min_r, img_height);
    assert_eq!(1 + max_c - min_c, img_width);
    // Вырезание результата из новой хромосомы
    (min_r..=max_r)
        .flat_map(|r| {
            (min_c..=max_c)
                .map(|c| new_chromosome[r * 2 * img_width + c])
                .collect::<Vec<_>>()
        })
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
        // Наименьшая оценка
        let min_dissimilarity = curr_gen_dissimilarities
            .iter()
            .cloned()
            .reduce(f32::min)
            .unwrap();
        // Наибольшая оценка
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
