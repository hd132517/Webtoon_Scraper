import os
import time
import cv2
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def get_episode_urls(webtoon_list_url, limit=3):
    """웹툰 목록 페이지에서 회차별 URL 자동 추출"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(webtoon_list_url)
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    episode_links = []
    for a in soup.select("a[href*='/webtoon/detail?titleId=']")[:limit]:
        href = a.get("href")
        title = a.get_text(strip=True).replace("/", "_").replace(" ", "_").partition("별점")[0].strip()
        url = f"https://comic.naver.com{href}"
        episode_links.append((url, title))

    return episode_links


def get_webtoon_images_with_selenium(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    viewer = soup.find("div", class_="wt_viewer")
    if not viewer:
        print("❌ 웹툰 이미지 영역을 찾지 못했습니다.")
        return 0

    img_tags = viewer.find_all("img")
    headers = {
        "Referer": "https://comic.naver.com",
        "User-Agent": "Mozilla/5.0"
    }

    count = 0
    for i, img in enumerate(img_tags):
        try:
            src = img.get("src")
            if not src:
                continue
            response = requests.get(src, headers=headers)
            if response.status_code == 200:
                filename = os.path.join(save_dir, f"{i:03}.jpg")
                with open(filename, "wb") as f:
                    f.write(response.content)
                count += 1
        except Exception as e:
            print(f"❌ 이미지 저장 실패: {e}")

    driver.quit()
    return count


def split_image_panels(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = sorted(os.listdir(input_folder))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 불러올 이미지가 없습니다.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 150 and h > 150:
                panel = img[y:y + h, x:x + w]
                filename = f"{os.path.splitext(img_name)[0]}_panel_{idx}.jpg"
                cv2.imwrite(os.path.join(output_folder, filename), panel)


if __name__ == "__main__":
    #list_url = input("🔗 나이트런 웹툰 목록 페이지 URL 입력:\n").strip()
    list_url = "https://comic.naver.com/webtoon/list?titleId=64997&page=1&sort=ASC"
    limit = int(input("📥 몇 회차까지 다운로드할까요? (예: 3): ").strip())

    episode_infos = get_episode_urls(list_url, limit=limit)
    if not episode_infos:
        print("❌ 회차 URL을 추출하지 못했습니다. 목록 페이지의 구조를 확인해주세요.")
    else:
        print(f"\n📄 총 {len(episode_infos)}개의 회차 URL 추출 완료.")

        for idx, (ep_url, ep_title) in enumerate(episode_infos):
            raw_dir = f"raw_images/{ep_title}"
            cut_dir = f"cut_panels/{ep_title}"

            print(f"\n📥 [{ep_title}] 이미지 다운로드 중...")
            count = get_webtoon_images_with_selenium(ep_url, save_dir=raw_dir)

            if count > 0:
                print(f"✂️ [{ep_title}] 이미지 컷 분할 중...")
                split_image_panels(input_folder=raw_dir, output_folder=cut_dir)

        print("\n✅ 전체 회차 다운로드 및 컷 분할 완료!")





