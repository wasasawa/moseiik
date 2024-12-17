#[cfg(test)]
mod tests {
    use image::RgbImage;
    use image::ImageReader;
    use std::error::Error;
    use moseiik::main::compute_mosaic;

    // Helper function to read an image and convert it to RgbImage
    fn to_rgb(image_path: &str) -> Result<RgbImage, Box<dyn Error>>  {
        let target = ImageReader::open(image_path)?.decode()?.into_rgb8();
        Ok(target)
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_x86() {
        // test avx2 or sse2 if available
            // Declaring args variable to recreate kit.jpeg with tiles from moseiik_test_images and 25 tile_size
    let args = moseiik::main::Options {
        image : "assets/kit.jpeg".to_string(),
        output : "tests/out.png".to_string(),
        tiles : "moseiik_test_images/images".to_string(),
        scaling : 1,
        tile_size : 25,
        remove_used : false,
        verbose : true,
        simd : false,
        num_thread : 1
    };

    // Recreating the image
    compute_mosaic(args);

    // Openning the generated image aka the output of compute mosaic and putting it in image_generated
    let image_generated : RgbImage;
    match to_rgb("tests/out.png") {
        Ok(image) => image_generated = image,
        Err(_) => panic!("Failed to load the output image in the generic test")
    }
    
    // Openning the reference image to compare with the generated image
    let image_ref : RgbImage;
    match to_rgb("assets/ground-truth-kit.png") {
        Ok(image) => image_ref = image,
        Err(_) => panic!("Couldnt load the provided image in the generic test")
    }

    // Testing if the images have the right width/height
    let (width_ref,height_ref) = image_generated.dimensions();
    let (width_gen,height_gen) = image_ref.dimensions();

    if width_gen != width_ref { panic!("The output image's width is not the same as the width of the provided image");}
    if height_gen != height_ref { panic!("The output image's height is not the same as the height of the provided image");}

    for x in 0..height_gen - 1 { 
        for y in 0..width_gen - 1 {
            let pixel_gen = image_generated.get_pixel(y,x);
            let pixel_ref = image_ref.get_pixel(y,x);

            if pixel_gen != pixel_ref {
                assert!(false,"The output image is different than the ref image in pixel w: {} ,h: {}", y, x);// The 2 images are not identical ?? maybe not used the same tiles ?
            }
        }
    }
        
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {
    // Declaring args variable to recreate kit.jpeg with tiles from moseiik_test_images and 25 tile_size
    let args = moseiik::main::Options {
        image : "assets/kit.jpeg".to_string(),
        output : "tests/out.png".to_string(),
        tiles : "moseiik_test_images/images".to_string(),
        scaling : 1,
        tile_size : 25,
        remove_used : false,
        verbose : true,
        simd : true,
        num_thread : 1
    };

    // Recreating the image
    compute_mosaic(args);

    // Openning the generated image aka the output of compute mosaic and putting it in image_generated
    let image_generated : RgbImage;
    match to_rgb("tests/out.png") {
        Ok(image) => image_generated = image,
        Err(_) => panic!("Failed to load the output image in the generic test")
    }
    
    // Openning the reference image to compare with the generated image
    let image_ref : RgbImage;
    match to_rgb("assets/ground-truth-kit.png") {
        Ok(image) => image_ref = image,
        Err(_) => panic!("Couldnt load the provided image in the generic test")
    }

    // Testing if the images have the right width/height
    let (width_ref,height_ref) = image_generated.dimensions();
    let (width_gen,height_gen) = image_ref.dimensions();

    if width_gen != width_ref { panic!("The output image's width is not the same as the width of the provided image");}
    if height_gen != height_ref { panic!("The output image's height is not the same as the height of the provided image");}

    for x in 0..height_gen - 1 { 
        for y in 0..width_gen - 1 {
            let pixel_gen = image_generated.get_pixel(y,x);
            let pixel_ref = image_ref.get_pixel(y,x);

            if pixel_gen != pixel_ref {
                assert!(false,"The output image is different than the ref image in pixel w: {} ,h: {}", y, x);
            }
        }
    }
    }

    #[test]
    // Integration test for the generic architecture
    fn test_generic() {
    // Declaring args variable to recreate kit.jpeg with tiles from moseiik_test_images and 25 tile_size
    let args = moseiik::main::Options {
        image : "assets/kit.jpeg".to_string(),
        output : "tests/out.png".to_string(),
        tiles : "moseiik_test_images/images".to_string(),
        scaling : 1,
        tile_size : 25,
        remove_used : false,
        verbose : true,
        simd : false,
        num_thread : 1
    };

    // Recreating the image
    compute_mosaic(args);

    // Openning the generated image aka the output of compute mosaic and putting it in image_generated
    let image_generated : RgbImage;
    match to_rgb("tests/out.png") {
        Ok(image) => image_generated = image,
        Err(_) => panic!("Failed to load the output image in the generic test")
    }
    
    // Openning the reference image to compare with the generated image
    let image_ref : RgbImage;
    match to_rgb("assets/ground-truth-kit.png") {
        Ok(image) => image_ref = image,
        Err(_) => panic!("Couldnt load the provided image in the generic test")
    }

    // Testing if the images have the right width/height
    let (width_ref,height_ref) = image_generated.dimensions();
    let (width_gen,height_gen) = image_ref.dimensions();

    if width_gen != width_ref { panic!("The output image's width is not the same as the width of the provided image");}
    if height_gen != height_ref { panic!("The output image's height is not the same as the height of the provided image");}

    for x in 0..height_gen - 1 { 
        for y in 0..width_gen - 1 {
            let pixel_gen = image_generated.get_pixel(y,x);
            let pixel_ref = image_ref.get_pixel(y,x);

            if pixel_gen != pixel_ref {
                assert!(false,"The output image is different than the ref image in pixel w: {} ,h: {}", y, x); 
            }
        }
    }
    }
}
