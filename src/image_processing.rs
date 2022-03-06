use iced::image::Handle;
use image::{GenericImage, GenericImageView, ImageBuffer, Pixel, Rgba, RgbaImage};
use lab::Lab;

use crate::genetic_algorithm::Chromosome;

pub fn get_image_handle(image: &RgbaImage) -> Handle {
    // RGBA -> BGRA
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

pub fn get_chromosome_image(
    image: &RgbaImage,
    piece_size: u32,
    chromosome: &Chromosome,
    show_incorrect: bool,
) -> RgbaImage {
    let pieces: Vec<Vec<_>> = (0..(image.height() / piece_size))
        .map(|r| {
            (0..(image.width() / piece_size))
                .map(|c| image.view(c * piece_size, r * piece_size, piece_size, piece_size))
                .collect()
        })
        .collect();

    let mut new_image = ImageBuffer::new(image.width(), image.height());
    for (r, v) in chromosome.iter().enumerate() {
        for (c, (i, j)) in v.iter().enumerate() {
            new_image
                .copy_from(
                    &*pieces[*i][*j],
                    (c as u32) * piece_size,
                    (r as u32) * piece_size,
                )
                .unwrap();
            if show_incorrect && (r != *i || c != *j) {
                for img_r in ((r as u32) * piece_size)..(((r + 1) as u32) * piece_size) {
                    new_image.put_pixel(
                        (c as u32) * piece_size,
                        img_r,
                        Rgba::from([255u8, 0, 0, 255]),
                    );
                    new_image.put_pixel(
                        ((c + 1) as u32) * piece_size - 1,
                        img_r,
                        Rgba::from([255u8, 0, 0, 255]),
                    );
                }
                for img_c in ((c as u32) * piece_size)..(((c + 1) as u32) * piece_size) {
                    new_image.put_pixel(
                        img_c,
                        (r as u32) * piece_size,
                        Rgba::from([255u8, 0, 0, 255]),
                    );
                    new_image.put_pixel(
                        img_c,
                        ((r + 1) as u32) * piece_size - 1,
                        Rgba::from([255u8, 0, 0, 255]),
                    );
                }
            }
        }
    }
    new_image
}

pub fn get_lab_image(image: &RgbaImage) -> Vec<Lab> {
    let rgb_pixels: Vec<_> = image
        .pixels()
        .flat_map(|rgba_pixel| rgba_pixel.to_rgb().channels().to_vec())
        .collect();
    lab::rgb_bytes_to_labs(&rgb_pixels)
}

pub fn image_direct_comparison(chromosome: &Chromosome) -> f32 {
    (chromosome
        .iter()
        .enumerate()
        .map(|(r, v)| {
            v.iter()
                .enumerate()
                .filter(|(c, ij)| (r, *c) == **ij)
                .count()
        })
        .sum::<usize>() as f32)
        * 100.0
        / ((chromosome.len() * chromosome[0].len()) as f32)
}

pub fn image_neighbour_comparison(chromosome: &Chromosome) -> f32 {
    let (img_height, img_width) = (chromosome.len(), chromosome[0].len());
    (0..img_height)
        .map(|r| {
            (0..img_width)
                .map(|c| {
                    let (mut res, mut count) = (0usize, 0usize);
                    let (i, j) = chromosome[r][c];
                    for (dr, dc) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                        let (new_r, new_c) = ((r as isize) + dr, (c as isize) + dc);
                        if new_r < 0 || new_c < 0 {
                            continue;
                        }
                        let (new_r, new_c) = (new_r as usize, new_c as usize);
                        if new_r >= img_height || new_c >= img_width {
                            continue;
                        }
                        count += 1;

                        let (new_i, new_j) = ((i as isize) + dr, (j as isize) + dc);
                        if new_i < 0 || new_j < 0 {
                            continue;
                        }
                        let (new_i, new_j) = (new_i as usize, new_j as usize);
                        if new_i >= img_height || new_j >= img_width {
                            continue;
                        }

                        if chromosome[new_r][new_c] == (new_i, new_j) {
                            res += 1;
                        }
                    }
                    (res as f32) / (count as f32)
                })
                .sum::<f32>()
        })
        .sum::<f32>()
        * 100.0
        / ((img_width * img_height) as f32)
}
