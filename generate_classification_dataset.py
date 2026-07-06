import os
import glob
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def clean_dir(path):
    import time
    for _ in range(5):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            return
        except Exception:
            time.sleep(0.5)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def main():
    base_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage"
    dataset_src = os.path.join(base_dir, "mar_258class_dataset", "final_dataset", "Flag_Training_Project")
    yaml_path = os.path.join(dataset_src, "dataset.yaml")
    dst_dir = os.path.join(base_dir, "kaggle_classification_dataset")
    full_neg_dir = os.path.join(base_dir, "full_frame_negatives")
    ref_country_dir = os.path.join(base_dir, "reference", "country")
    ref_inst_dir = os.path.join(base_dir, "reference", "institution")

    # 1. Embedded class names (254 flag classes)
    names = {0: 'International_Olympic_Committee', 1: 'NATO', 2: 'UNESCO', 3: 'United_Nations', 4: 'ad', 5: 'ae', 6: 'af', 7: 'ag', 8: 'ai', 9: 'al', 10: 'am', 11: 'ao', 12: 'aq', 13: 'ar', 14: 'as', 15: 'at', 16: 'au', 17: 'aw', 18: 'ax', 19: 'az', 20: 'ba', 21: 'bb', 22: 'bd', 23: 'be', 24: 'bf', 25: 'bg', 26: 'bh', 27: 'bi', 28: 'bj', 29: 'bl', 30: 'bm', 31: 'bn', 32: 'bo', 33: 'bq', 34: 'br', 35: 'bs', 36: 'bt', 37: 'bv', 38: 'bw', 39: 'by', 40: 'bz', 41: 'ca', 42: 'cc', 43: 'cd', 44: 'cf', 45: 'cg', 46: 'ch', 47: 'ci', 48: 'ck', 49: 'cl', 50: 'cm', 51: 'cn', 52: 'co', 53: 'cr', 54: 'cu', 55: 'cv', 56: 'cw', 57: 'cx', 58: 'cy', 59: 'cz', 60: 'de', 61: 'dj', 62: 'dk', 63: 'dm', 64: 'do', 65: 'dz', 66: 'ec', 67: 'ee', 68: 'eg', 69: 'eh', 70: 'er', 71: 'es', 72: 'et', 73: 'fi', 74: 'fj', 75: 'fk', 76: 'fm', 77: 'fo', 78: 'fr', 79: 'ga', 80: 'gb', 81: 'gb-eng', 82: 'gb-nir', 83: 'gb-sct', 84: 'gb-wls', 85: 'gd', 86: 'ge', 87: 'gf', 88: 'gg', 89: 'gh', 90: 'gi', 91: 'gl', 92: 'gm', 93: 'gn', 94: 'gp', 95: 'gq', 96: 'gr', 97: 'gs', 98: 'gt', 99: 'gu', 100: 'gw', 101: 'gy', 102: 'hk', 103: 'hm', 104: 'hn', 105: 'hr', 106: 'ht', 107: 'hu', 108: 'id', 109: 'ie', 110: 'il', 111: 'im', 112: 'in', 113: 'io', 114: 'iq', 115: 'ir', 116: 'is', 117: 'it', 118: 'je', 119: 'jm', 120: 'jo', 121: 'jp', 122: 'ke', 123: 'kg', 124: 'kh', 125: 'ki', 126: 'km', 127: 'kn', 128: 'kp', 129: 'kr', 130: 'kw', 131: 'ky', 132: 'kz', 133: 'la', 134: 'lb', 135: 'lc', 136: 'li', 137: 'lk', 138: 'lr', 139: 'ls', 140: 'lt', 141: 'lu', 142: 'lv', 143: 'ly', 144: 'ma', 145: 'mc', 146: 'md', 147: 'me', 148: 'mf', 149: 'mg', 150: 'mh', 151: 'mk', 152: 'ml', 153: 'mm', 154: 'mn', 155: 'mo', 156: 'mp', 157: 'mq', 158: 'mr', 159: 'ms', 160: 'mt', 161: 'mtc', 162: 'mu', 163: 'mv', 164: 'mw', 165: 'mx', 166: 'my', 167: 'mz', 168: 'na', 169: 'nc', 170: 'ne', 171: 'nf', 172: 'ng', 173: 'ni', 174: 'nl', 175: 'no', 176: 'np', 177: 'nr', 178: 'nu', 179: 'nz', 180: 'om', 181: 'pa', 182: 'pe', 183: 'pf', 184: 'pg', 185: 'ph', 186: 'pk', 187: 'pl', 188: 'pm', 189: 'pn', 190: 'pr', 191: 'ps', 192: 'pt', 193: 'pw', 194: 'py', 195: 'qa', 196: 're', 197: 'ro', 198: 'rs', 199: 'ru', 200: 'rw', 201: 'sa', 202: 'sb', 203: 'sd', 204: 'se', 205: 'sg', 206: 'sh', 207: 'si', 208: 'sj', 209: 'sk', 210: 'sl', 211: 'sm', 212: 'sn', 213: 'so', 214: 'sr', 215: 'ss', 216: 'st', 217: 'sv', 218: 'sx', 219: 'sy', 220: 'sz', 221: 'tc', 222: 'td', 223: 'tg', 224: 'th', 225: 'tj', 226: 'tk', 227: 'tl', 228: 'tm', 229: 'tn', 230: 'to', 231: 'tr', 232: 'tt', 233: 'tw', 234: 'tz', 235: 'ua', 236: 'ug', 237: 'us', 238: 'uy', 239: 'uz', 240: 'va', 241: 'vc', 242: 've', 243: 'vi', 244: 'vn', 245: 'vu', 246: 'wf', 247: 'ws', 248: 'xk', 249: 'ye', 250: 'yt', 251: 'za', 252: 'zm', 253: 'zw'}

    # 2. Map class names to template paths
    template_paths = {}
    for path in glob.glob(os.path.join(ref_country_dir, "*")):
        if path.lower().endswith(".png"):
            name = os.path.splitext(os.path.basename(path))[0]
            template_paths[name] = path
    for path in glob.glob(os.path.join(ref_inst_dir, "*")):
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            name = os.path.splitext(os.path.basename(path))[0]
            template_paths[name] = path

    class_to_template = {}
    missing_templates = []
    for cls_id, cls_name in names.items():
        if cls_name in template_paths:
            class_to_template[cls_name] = template_paths[cls_name]
        else:
            missing_templates.append(cls_name)

    if missing_templates:
        print(f"[WARNING] Missing template files for classes: {missing_templates}")
    else:
        print("All templates successfully mapped to class names.")

    # 3. Load negative background images
    # We collect all negative images from the original dataset and full_frame_negatives
    dataset_neg_files = []
    all_jpgs = glob.glob(os.path.join(dataset_src, "*.jpg"))
    for img_path in all_jpgs:
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            if not lines or not any(l.strip() for l in lines):
                dataset_neg_files.append(img_path)
        else:
            dataset_neg_files.append(img_path)

    all_negatives = list(dataset_neg_files)
    if os.path.exists(full_neg_dir):
        neg_files_extra = glob.glob(os.path.join(full_neg_dir, "*.jpg"))
        all_negatives.extend(neg_files_extra)
        print(f"Loaded {len(neg_files_extra)} extra negatives.")

    print(f"Total negative images loaded: {len(all_negatives)}")
    if not all_negatives:
        print("[ERROR] No background negative images found!")
        return

    # 4. Initialize directories
    print("\n=== Initializing classification dataset folders ===")
    for split in ['train', 'val']:
        clean_dir(os.path.join(dst_dir, split))
        for class_name in names.values():
            os.makedirs(os.path.join(dst_dir, split, class_name), exist_ok=True)
        os.makedirs(os.path.join(dst_dir, split, "background"), exist_ok=True)

    # 5. Helper function to generate a single crop
    def generate_synthetic_crop(template_path, bg_images, size=128):
        # 1. Crop background patch
        bg_img = None
        for _ in range(10):
            try:
                bg_path = random.choice(bg_images)
                with Image.open(bg_path) as bi:
                    bw_img, bh_img = bi.size
                    if bw_img >= size and bh_img >= size:
                        x0 = random.randint(0, bw_img - size)
                        y0 = random.randint(0, bh_img - size)
                        bg_img = bi.crop((x0, y0, x0 + size, y0 + size)).copy()
                        break
            except Exception:
                continue
        if bg_img is None:
            # Fallback background
            bg_img = Image.new('RGB', (size, size), (218, 187, 156))

        # 2. Load and augment flag
        try:
            with Image.open(template_path) as flag_img:
                flag = flag_img.convert('RGBA')
                orig_w, orig_h = flag.size

                # Scale flag randomly
                tw = random.randint(45, 90)
                th = max(5, int((tw / orig_w) * orig_h))
                flag = flag.resize((tw, th), Image.Resampling.LANCZOS)

                # Rotate flag randomly
                angle = random.uniform(-20, 20)
                flag = flag.rotate(angle, expand=True)

                # Paste flag on background at center + offset
                fw, fh = flag.size
                cx = size // 2 - fw // 2
                cy = size // 2 - fh // 2
                dx = random.randint(-15, 15)
                dy = random.randint(-15, 15)

                bg_img.paste(flag, (cx + dx, cy + dy), flag)
        except Exception as e:
            # If template fails, return background as-is (graceful degradation)
            pass

        # 3. Apply lens effects (brightness, contrast, color, blur)
        bg_img = ImageEnhance.Color(bg_img).enhance(random.uniform(0.85, 1.25))
        bg_img = ImageEnhance.Brightness(bg_img).enhance(random.uniform(0.85, 1.15))
        bg_img = ImageEnhance.Contrast(bg_img).enhance(random.uniform(0.85, 1.20))
        
        if random.random() < 0.30:
            bg_img = bg_img.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.4)))
            
        return bg_img.convert('RGB')

    # 6. Generate synthetic images for all 254 classes
    random.seed(42)
    
    num_train_per_class = 60
    num_val_per_class = 15
    
    print("\n=== Generating synthetic flag classification dataset ===")
    for cls_name, t_path in class_to_template.items():
        # Train split
        for i in range(num_train_per_class):
            crop = generate_synthetic_crop(t_path, all_negatives, size=128)
            dst_path = os.path.join(dst_dir, "train", cls_name, f"synth_{cls_name}_{i:03d}.jpg")
            crop.save(dst_path, 'JPEG', quality=95)
            
        # Val split
        for i in range(num_val_per_class):
            crop = generate_synthetic_crop(t_path, all_negatives, size=128)
            dst_path = os.path.join(dst_dir, "val", cls_name, f"synth_{cls_name}_{i:03d}.jpg")
            crop.save(dst_path, 'JPEG', quality=95)

    print(f"Generated {num_train_per_class} train and {num_val_per_class} val crops per class.")

    # 7. Generate background class crops
    print("\n=== Generating background/negative class crops ===")
    num_train_bg = 150
    num_val_bg = 40

    # Train background
    for i in range(num_train_bg):
        crop = generate_synthetic_crop(None, all_negatives, size=128)
        dst_path = os.path.join(dst_dir, "train", "background", f"bg_patch_{i:04d}.jpg")
        crop.save(dst_path, 'JPEG', quality=90)

    # Val background
    for i in range(num_val_bg):
        crop = generate_synthetic_crop(None, all_negatives, size=128)
        dst_path = os.path.join(dst_dir, "val", "background", f"bg_patch_{i:04d}.jpg")
        crop.save(dst_path, 'JPEG', quality=90)

    print(f"Generated {num_train_bg} train background crops and {num_val_bg} val background crops.")
    print(f"\n=== Synthetic classification dataset generation complete! Saved under {dst_dir} ===")

if __name__ == '__main__':
    main()
