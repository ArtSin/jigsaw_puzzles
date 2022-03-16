use iced::image::Handle;
use image::{GenericImage, GenericImageView, ImageBuffer, Pixel, Rgba, RgbaImage};
use lab::Lab;

use crate::{Solution, NEIGHBOUR_DIRECTIONS};

// Получение указателя (Handle) на изображение для iced
pub fn get_image_handle(image: &RgbaImage) -> Handle {
    // Преобразование из RGBA в BGRA
    Handle::from_pixels(
        image.width(),
        image.height(),
        image
            .as_raw()
            .chunks_exact(4)
            .flat_map(|arr| [arr[2], arr[1], arr[0], arr[3]])
            .collect(),
    )
}

// Получение изображения решения и отображение неправильных деталей
pub fn get_solution_image(
    image: &RgbaImage,
    piece_size: u32,
    img_width: usize,
    img_height: usize,
    solution: &Solution,
    show_incorrect: bool,
    show_neighbour: bool,
) -> RgbaImage {
    // Изображения каждой детали
    let pieces: Vec<Vec<_>> = (0..(img_height as u32))
        .map(|r| {
            (0..(img_width as u32))
                .map(|c| image.view(c * piece_size, r * piece_size, piece_size, piece_size))
                .collect()
        })
        .collect();

    // Изображение решения
    let mut new_image = ImageBuffer::new(image.width(), image.height());
    for r in 0..img_height {
        for c in 0..img_width {
            // Деталь в заданной позиции
            let (i, j) = &solution[r * img_width + c];
            // Копирование изображения детали в нужную позицию
            new_image
                .copy_from(
                    &*pieces[*i][*j],
                    (c as u32) * piece_size,
                    (r as u32) * piece_size,
                )
                .unwrap();

            // Если требуется отображение неправильных деталей
            if show_incorrect {
                let (r32, c32) = (r as u32, c as u32);
                // Диапазоны пикселей каждой стороны детали (верх, низ, лево, право)
                let dirs_ranges = [
                    (
                        (r32 * piece_size)..=(r32 * piece_size),
                        (c32 * piece_size)..=((c32 + 1) * piece_size - 1),
                    ),
                    (
                        ((r32 + 1) * piece_size - 1)..=((r32 + 1) * piece_size - 1),
                        (c32 * piece_size)..=((c32 + 1) * piece_size - 1),
                    ),
                    (
                        (r32 * piece_size)..=((r32 + 1) * piece_size - 1),
                        (c32 * piece_size)..=(c32 * piece_size),
                    ),
                    (
                        (r32 * piece_size)..=((r32 + 1) * piece_size - 1),
                        ((c32 + 1) * piece_size - 1)..=((c32 + 1) * piece_size - 1),
                    ),
                ];

                // Прямое сравнение
                if !show_neighbour && (r != *i || c != *j) {
                    // Закрашивание всех сторон детали
                    for (r_range, c_range) in &dirs_ranges {
                        for img_r in r_range.clone() {
                            for img_c in c_range.clone() {
                                new_image.put_pixel(img_c, img_r, Rgba::from([255u8, 0, 0, 255]));
                            }
                        }
                    }
                }
                // Сравнение по соседям
                else if show_neighbour {
                    // Перебор соседей
                    for (ind, (dr, dc)) in NEIGHBOUR_DIRECTIONS.iter().enumerate() {
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

                        // Сосед детали в исходном изображении
                        let (new_i, new_j) = ((*i as isize) + dr, (*j as isize) + dc);
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
                            continue;
                        }

                        // Если сосед неправильный, то сторона закрашивается
                        let (r_range, c_range) = &dirs_ranges[ind];
                        for img_r in r_range.clone() {
                            for img_c in c_range.clone() {
                                new_image.put_pixel(img_c, img_r, Rgba::from([255u8, 0, 0, 255]));
                            }
                        }
                    }
                }
            }
        }
    }
    new_image
}

// Преобразование изображения в цветовое пространство L*a*b*
pub fn get_lab_image(image: &RgbaImage) -> Vec<Lab> {
    let rgb_pixels: Vec<_> = image
        .pixels()
        .flat_map(|rgba_pixel| rgba_pixel.to_rgb().channels().to_vec())
        .collect();
    lab::rgb_bytes_to_labs(&rgb_pixels)
}
