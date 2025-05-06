import os
import requests
from serpapi import GoogleSearch
from PIL import Image
from io import BytesIO

# Set up API Key for SerpAPI (Google/Bing Images API)
SERP_API_KEY = os.environ["SERP_API_KEY"]

# Query for dirty dishes images
queries = [
    "dirty saute pan with burnt residue",
    "dirty casserole with food residue and food stains",
    "dirty dutch oven with food residue and food stains",
    "dirty wok with food residue and food stains",
    "dirty sauce pan with sauce residue"
]

# Output directory
output_dir = "dataset/dirty_dishes"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=False)


def download_image(url, filepath):
    """Download image from URL and save."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")  # Convert to RGB format
            image = image.resize((512, 512))  # Resize for consistency
            image.save(filepath, "JPEG")
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False


import time


def scrape_images(query, max_images=500):
    """Scrape images from Google/Bing using SerpAPI with rate limit handling."""
    count = 0
    start = 0  # Start index for pagination

    while count < max_images:
        search = GoogleSearch(
            {
                "q": query,
                "tbm": "isch",
                "api_key": SERP_API_KEY,
                "num": 100,  # SerpAPI returns max 100 per request
                "start": start,  # Use pagination to get more images
            }
        )

        results = search.get_dict()
        images = results.get("images_results", [])

        if not images:
            print(f"🚨 No more images found for '{query}'. Stopping.")
            break

        for img in images:
            url = img.get("original")
            if url:
                filepath = os.path.join(
                    output_dir, f"{query.replace(' ', '_')}_{count}.jpg"
                )
                if download_image(url, filepath):
                    count += 1
                if count >= max_images:
                    break

        print(f"🔄 Downloaded {count} images so far for '{query}'")

        start += 100  # Get next batch of 100 images
        time.sleep(5)  # Wait 5 seconds to avoid hitting rate limits

    print(f"✅ Finished downloading {count} images for '{query}'")


# Run scraping for all queries
for query in queries:
    scrape_images(query, max_images=500)  # Get 100 images per query

print("✅ Dataset collection complete! Images saved in dataset/dirty_dishes")
