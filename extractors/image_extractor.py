import fitz  # PyMuPDF
import os
import json

def extract_images_with_context(pdf_path: str, output_dir: str = "images_output"):
    """
    Extracts all visible images from a PDF, combines them with soft masks (if any),
    captures the nearest paragraph below each image as context, and saves results to JSON.

    Args:
        pdf_path (str): Path to the input PDF.
        output_dir (str): Directory to save images and JSON output.

    Returns:
        tuple: (json_path, image_context_data)
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_context_data = []

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_no, page in enumerate(doc, start=1):
        images = page.get_images(full=True)
        text_blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no, ...)
        text_blocks = sorted(text_blocks, key=lambda b: b[1])  # sort by y0 (vertical position)

        for img_no, img in enumerate(images, start=1):
            xref = img[0]
            smask = img[1]
            bbox = img[1:5]  # (x0, y0, x1, y1)
            img_bottom = bbox[3]

            try:
                # Create main pixmap
                main_pix = fitz.Pixmap(doc, xref)

                # ✅ Combine mask if available
                if smask != 0:
                    mask_pix = fitz.Pixmap(doc, smask)
                    if mask_pix.alpha or mask_pix.colorspace.n == 1:
                        main_pix = fitz.Pixmap(main_pix, mask_pix)
                    mask_pix = None
                else:
                    if main_pix.colorspace is None or main_pix.n < 3:
                        continue

                # Save image
                img_filename = f"{pdf_name}_page{page_no}_img{img_no}.png"
                img_path = os.path.join(output_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(main_pix.tobytes("png"))

                # 🧠 Find nearest paragraph *below* the image
                below_paras = [
                    t for t in text_blocks if t[1] > img_bottom
                ]  # t[1] = y0 of text block

                context_text = ""
                if below_paras:
                    # choose the *closest* block (minimum vertical distance)
                    closest_para = min(below_paras, key=lambda t: t[1] - img_bottom)
                    context_text = closest_para[4].strip()

                image_context_data.append({
                    "page": page_no,
                    "image_path": img_path,
                    "bbox": bbox,
                    "context": context_text if context_text else "No nearby text found."
                })

                main_pix = None  # free memory

            except Exception as e:
                print(f"Error processing page {page_no}, image {img_no}: {e}")

    # Save metadata as JSON
    json_path = os.path.join(output_dir, f"{pdf_name}_images.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(image_context_data, f, indent=4, ensure_ascii=False)

    return json_path, image_context_data
