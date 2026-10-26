"""
Sample dataset generator.
Creates a rich, diverse, balanced labeled dataset for model training and demo
across e-commerce, consumer electronics, media, and everyday services.
"""

import pandas as pd
import random
from typing import List

# ── Seed High-Quality Reviews ──────────────────────────────────────────────────
POSITIVE_REVIEWS = [
    "This product is absolutely amazing, exceeded all my expectations.",
    "Fantastic quality, fast delivery, and great customer service.",
    "I love this item so much, will definitely buy again!",
    "Best purchase I have made this year, highly recommended.",
    "Outstanding performance and brilliant build quality.",
    "The design is beautiful and quality is top notch.",
    "Absolutely loved it! Works perfectly and looks great.",
    "Exceeded expectations in every way possible, truly wonderful.",
    "Superb quality and arrived really quickly, very happy.",
    "Five stars! Perfect product, no complaints at all.",
    "Brilliant product, works exactly as described and more.",
    "Really impressed with the quality and fast shipping.",
    "Wonderful experience, the product is even better in person.",
    "This is hands down the best product I have ever bought.",
    "Amazing value for money, definitely worth every penny.",
    "Great product, easy to use and delivers excellent results.",
    "Very satisfied with this purchase, performs beyond expectations.",
    "Impressive quality and design, I am absolutely delighted.",
    "Outstanding product! Delivery was fast and packaging excellent.",
    "Incredible value, superb quality, great customer support.",
    "Love this! Works great and looks even better in real life.",
    "Very happy with this purchase. Quality is excellent.",
    "This exceeded all my expectations. Will recommend to friends.",
    "Perfect product. Does exactly what it promises and more.",
    "Brilliant item. Arrived fast, packaged well, works great.",
    "Really happy with this. The quality is absolutely top notch.",
    "Fantastic buy. So glad I purchased this item.",
    "The product is stunning and the quality is phenomenal.",
    "I am very impressed. Excellent quality and great price.",
    "Best product ever! I could not be more satisfied.",
    "Loved every aspect of this. Delivery was super fast too.",
    "Wonderful product, works flawlessly and looks premium.",
    "This is brilliant. So pleased with the quality and finish.",
    "Absolutely delighted. Five star product with zero issues.",
    "Top quality product. Works better than I expected.",
    "Amazing! Packaged perfectly, works great, very impressed.",
    "This is genuinely brilliant. Best money I have spent.",
    "Superb quality and looks even nicer than the photos show.",
    "Very happy with this item. It exceeded all expectations.",
    "Fantastic product. Will definitely be buying again soon.",
    "The display is bright and colors are vivid, love using this device.",
    "Camera takes sharp photos even in low light, very satisfied.",
    "Software runs fast without noticeable lag or freezing.",
    "Comfortable fit and high quality materials, feels premium.",
    "Audio is crystal clear with deep bass and no distortion.",
    "Battery life easily lasts two full days on a single charge.",
    "Setup guide was straightforward and took less than ten minutes.",
    "The app interface is clean, intuitive, and feature rich.",
    "Exquisite design with smooth curves and pleasant tactile feel.",
    "Remarkably quiet operation even under heavy computational load.",
]

NEGATIVE_REVIEWS = [
    "This product is absolutely terrible, complete waste of money.",
    "Broken on arrival, very poor quality, will not buy again.",
    "Worst purchase ever, totally disappointed with this item.",
    "Product stopped working after just two days, very poor.",
    "Extremely poor quality, nothing like the description says.",
    "Do not buy this, the quality is awful and it broke quickly.",
    "Very disappointed, cheaply made and feels flimsy.",
    "Complete rubbish, does not work at all, waste of money.",
    "Terrible product, returned immediately and very unhappy.",
    "Absolutely dreadful quality, nothing like the pictures shown.",
    "Do not waste your money on this awful product.",
    "Stopped working within a week, extremely poor quality.",
    "Very bad product, totally not worth the price at all.",
    "Rubbish build quality, broke the first time I used it.",
    "Shocking quality, nothing like what was advertised.",
    "Would not recommend this, cheap material and poorly made.",
    "Complete waste of money, very bad customer service too.",
    "Terrible experience, the product was defective on arrival.",
    "Very poor quality control, product was broken out of the box.",
    "Awful product, I deeply regret purchasing this item.",
    "Not happy at all, this product does not function correctly.",
    "Terrible quality, feels cheap and fell apart very quickly.",
    "Extremely disappointed with this purchase, very poor value.",
    "This is the worst product I have ever bought, avoid it.",
    "Dreadful quality and the after sales service was awful too.",
    "Massive disappointment. Stopped working after one use.",
    "Cheap junk. Nothing like described. Avoid at all costs.",
    "Bad quality, bad delivery, bad everything. Do not buy.",
    "Absolute rubbish. Overpriced for such appalling quality.",
    "Would give zero stars if I could. Total waste of money.",
    "Battery drains completely in under an hour, practically unusable.",
    "Customer service was rude and refused to process my return.",
    "App constantly crashes whenever I try to open my profile.",
    "Missing essential parts in the box, very frustrating experience.",
    "The screen flickers uncontrollably after fifteen minutes of usage.",
    "Shipping took three weeks and arrived in a severely damaged box.",
    "Loud rattling noise whenever the motor runs, awful engineering.",
    "Overheats dangerously after short usage, do not buy this hazard.",
    "Material felt like cheap rough plastic with an awful chemical smell.",
    "Buttons stick constantly and fail to register half my clicks.",
    "Completely useless device that fails to perform the simplest tasks.",
    "Returned within twenty four hours due to defective components.",
    "False advertising, does not possess the features claimed on the box.",
    "Terrible software full of bugs, glitches, and unexpected crashes.",
    "Fell apart into multiple pieces during the second week of use.",
]

NEUTRAL_REVIEWS = [
    "The product arrived on the expected delivery date.",
    "This item is the standard model with basic features included.",
    "The package was delivered this morning without any issues.",
    "The product has the features that are listed in the description.",
    "The item was released in the latest product update cycle.",
    "The model comes with one year standard manufacturer warranty.",
    "This product is available in three different color options.",
    "The item measures approximately 20 centimeters in length.",
    "The device operates on a standard USB power connection.",
    "The product ships from the warehouse in three to five days.",
    "This model was updated with the new firmware last month.",
    "The product is compatible with all standard operating systems.",
    "The item weighs approximately 500 grams including the packaging.",
    "The software update was released yesterday for this device.",
    "This product is manufactured in the standard production facility.",
    "The item comes in standard packaging with the user manual.",
    "The product is listed as available on the company website.",
    "This device connects using the standard Bluetooth protocol.",
    "The item requires two AA batteries for standard operation.",
    "This product dimensions are listed on the product specification page.",
    "The phone was announced at the technology conference yesterday.",
    "The new model was launched in the first quarter of this year.",
    "The specifications of this device were published on the website.",
    "The company released a statement about its new product range.",
    "The item can be found in stores from next week onwards.",
    "The product was reviewed in the latest technology magazine.",
    "The device has a charging time of approximately two hours.",
    "This product is listed under the electronics category online.",
    "The item is currently available for purchase on the main website.",
    "The product details are available on the manufacturer page.",
    "The shipment contains one unit along with standard accessory cables.",
    "The exterior housing is constructed using standard composite polymer.",
    "The unit draws approximately forty watts during normal operation.",
    "Standard delivery requires between three and five business days.",
    "The replacement filter is available through authorized retail partners.",
    "Serial numbers are embossed onto the rear plate of each chassis.",
    "The technical specification sheet was updated in the latest revision.",
    "This hardware revision incorporates standard regulatory compliance chips.",
    "Assembly requires standard Phillips screwdriver according to the manual.",
    "The catalog listing includes dimensions and gross package weight.",
]

# ── Dynamic Templates for Generating Distinct Synthesized Records ───────────
DOMAINS = [
    "laptop", "monitor", "keyboard", "headphones", "smartwatch",
    "tablet", "camera", "speaker", "vacuum cleaner", "blender",
    "coffee maker", "microwave", "toaster", "backpack", "jacket",
    "office chair", "running shoes", "air purifier", "water bottle",
    "desk lamp", "wireless charger", "external drive", "fitness band",
    "earbuds", "router", "game controller", "electric kettle"
]

POS_ADJS = [
    "reliable", "impressive", "solid", "delightful", "superb",
    "sturdy", "convenient", "sleek", "high performance", "well made",
    "efficient", "versatile", "responsive", "intuitive", "top tier"
]
POS_VERBS = [
    "works smoothly", "exceeded my expectations", "feels well built",
    "delivers clear quality", "saves time every day", "operates without issues",
    "handles heavy usage easily", "runs whisper quiet", "looks stunning on desk"
]
POS_CLOSERS = [
    "Worth every dollar spent.", "Very happy with the overall purchase.",
    "Would recommend to friends.", "Arrived fast and in good shape.",
    "A solid five star addition.", "Pleasantly surprised by this item.",
    "Definitely a great addition to my setup.", "Great build and great support."
]

NEG_ADJS = [
    "flimsy", "disappointing", "unreliable", "defective", "subpar",
    "poorly constructed", "frustrating", "faulty", "clunky", "cheaply made",
    "unresponsive", "inconsistent", "fragile", "shoddy", "glitchy"
]
NEG_VERBS = [
    "broke after limited use", "stopped working unexpectedly",
    "failed to perform as advertised", "causes frequent headaches",
    "feels like cheap imitation", "has constant connectivity drops",
    "drains battery far too quickly", "overheats after brief operation"
]
NEG_CLOSERS = [
    "Definitely seeking a full refund.", "Would not recommend this to anyone.",
    "Completely wasted my money here.", "Extremely unhappy with this order.",
    "Avoid this product completely.", "Customer support was unhelpful.",
    "Returned within forty eight hours.", "Regrettable purchase all around."
]

NEU_VERBS = [
    "operates within standard thermal limits",
    "was packaged according to standard logistics protocols",
    "includes standard two meter cable",
    "conforms to industry safety specifications",
    "is documented in the manufacturer handbook",
    "carries the standard CE certification mark",
    "shipped through standard ground postal courier",
    "matches the technical catalogue serial number",
    "measures within the expected dimension tolerances",
    "uses standard AC power input"
]
NEU_CLOSERS = [
    "The user guide provides standard operating instructions.",
    "Additional technical parameters are listed on the box.",
    "Warranty information is enclosed with the receipt.",
    "Specifications remain unchanged from the prior release.",
    "The product complies with regional regulatory standards.",
    "Maintenance requirements are outlined in the leaflet.",
    "The item is categorized under general consumer hardware."
]

# Nuanced / Mixed Sentiment Sentences (Realistic real-world edge cases)
MIXED_POS = [
    "The screen is bright although the speakers are only average, still a good purchase.",
    "Delivery was slightly delayed but the device performance makes up for it completely.",
    "A bit heavy to carry around, however build quality is very solid and reliable.",
    "Software has a few small quirks though overall it handles all my tasks wonderfully.",
    "Price is somewhat steep, but the durability and features make it worth buying.",
]

MIXED_NEG = [
    "The design looks decent on the outside but the internal hardware failed quickly.",
    "Customer service was polite yet they completely failed to resolve my issue.",
    "Packaging was nice but the product inside was chipped and defective.",
    "Setup was easy enough, but it stopped turning on after only four days.",
    "Good concept in theory, but execution is terribly flawed and frustrating to use.",
]

MIXED_NEU = [
    "The product has some positive reviews online while others report average results.",
    "Initial feedback from users varies according to recent market survey reports.",
    "The manufacturer website lists both standard features and optional accessories.",
    "The technical review outlines both the functional aspects and basic dimensions.",
    "The retail price reflects standard market positioning for this hardware class.",
]


def generate_sample_dataset(n_per_class: int = 200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a balanced, diverse labeled dataset with unique cleaned phrases.
    Guarantees no duplication collapse after NLP preprocessing.
    """
    random.seed(random_seed)

    pos_pool = list(POSITIVE_REVIEWS) + list(MIXED_POS)
    neg_pool = list(NEGATIVE_REVIEWS) + list(MIXED_NEG)
    neu_pool = list(NEUTRAL_REVIEWS) + list(MIXED_NEU)

    # Combinatorial synthesis to ensure unique textual content
    seen_pos = set(pos_pool)
    while len(pos_pool) < n_per_class * 2:
        d = random.choice(DOMAINS)
        a = random.choice(POS_ADJS)
        v = random.choice(POS_VERBS)
        c = random.choice(POS_CLOSERS)
        sentence = f"The {d} is {a} and {v}. {c}"
        if sentence not in seen_pos:
            seen_pos.add(sentence)
            pos_pool.append(sentence)

    seen_neg = set(neg_pool)
    while len(neg_pool) < n_per_class * 2:
        d = random.choice(DOMAINS)
        a = random.choice(NEG_ADJS)
        v = random.choice(NEG_VERBS)
        c = random.choice(NEG_CLOSERS)
        sentence = f"The {d} is {a} and {v}. {c}"
        if sentence not in seen_neg:
            seen_neg.add(sentence)
            neg_pool.append(sentence)

    seen_neu = set(neu_pool)
    while len(neu_pool) < n_per_class * 2:
        d = random.choice(DOMAINS)
        v = random.choice(NEU_VERBS)
        c = random.choice(NEU_CLOSERS)
        sentence = f"The {d} {v}. {c}"
        if sentence not in seen_neu:
            seen_neu.add(sentence)
            neu_pool.append(sentence)

    # Sample exactly n_per_class unique records per category
    final_pos = random.sample(pos_pool, min(n_per_class, len(pos_pool)))
    final_neg = random.sample(neg_pool, min(n_per_class, len(neg_pool)))
    final_neu = random.sample(neu_pool, min(n_per_class, len(neu_pool)))

    records = (
        [{"text": t, "sentiment": "Positive"} for t in final_pos] +
        [{"text": t, "sentiment": "Negative"} for t in final_neg] +
        [{"text": t, "sentiment": "Neutral"}  for t in final_neu]
    )

    df = pd.DataFrame(records).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_sample_dataset(200)
    print(f"Generated {len(df)} records.")
    print(df["sentiment"].value_counts())
