import os
import time
import cv2
import requests
import base64
import numpy as np
import shutil
import math
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin


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
    seen = set()

    for a in soup.select("a[href*='/webtoon/detail?titleId=']"):
        href = a.get("href")
        title = a.get_text(strip=True).replace("/", "_").replace(" ", "_").partition("별점")[0].strip()
        full_url = urljoin("https://comic.naver.com", href)

        if full_url == "https://comic.naver.com/webtoon/detail?titleId=64997&no=1&week=sat":    # 예고편 제거
            continue
        if full_url not in seen:
            seen.add(full_url)
            episode_links.append((full_url, title))
        if len(episode_links) >= limit:
            break

    return episode_links


def get_webtoon_images_with_selenium(url, save_dir="raw_images"):
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
        driver.quit()
        return 0

    img_tags = viewer.find_all("img")
    if not img_tags:
        print("❌ 이미지 태그가 없습니다.")
        driver.quit()
        return 0

    headers = {
        "Referer": "https://comic.naver.com",
        "User-Agent": "Mozilla/5.0"
    }

    count = 0
    for i, img in enumerate(img_tags):
        src = img.get("src")
        if not src:
            continue
        try:
            response = requests.get(src, headers=headers)
            if response.status_code == 200:
                filename = os.path.join(save_dir, f"{i:03}.jpg")
                with open(filename, "wb") as f:
                    f.write(response.content)
                count += 1
        except Exception as e:
            print(f"❌ 이미지 다운로드 실패: {e}")

    driver.quit()
    print(f"✅ {count}개의 이미지 저장 완료.")
    return count


def detect_panel_ranges(binary_img, axis='vertical', threshold=250, min_size=50):
    """세로 or 가로 방향 기준으로 컷 구간 리스트 추출"""
    if axis == 'vertical':
        profile = np.mean(binary_img, axis=1)
        size = binary_img.shape[0]
    else:
        profile = np.mean(binary_img, axis=0)
        size = binary_img.shape[1]

    ranges = []
    in_panel = False
    start = 0
    for i in range(size):
        if not in_panel and profile[i] < threshold:
            in_panel = True
            start = i
        elif in_panel and profile[i] >= threshold:
            in_panel = False
            end = i
            if end - start > min_size:
                ranges.append((start, end))
    if in_panel and size - start > min_size:
        ranges.append((start, size))

    return ranges if ranges else [(0, size)]


def recursively_split_and_save(img, base_name, panel_idx, save_dir, depth=0, max_depth=3, threshold=252):
    """
    이미지 내 패널을 재귀적으로 분할하여 저장
    """
    if depth > max_depth:
        return 0, panel_idx

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)    # 가우시안 블러
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    v_ranges = detect_panel_ranges(binary, axis='vertical', threshold=threshold)

    total_saved = 0
    for y1, y2 in v_ranges:
        v_crop = img[y1:y2, :]

        h_gray = cv2.cvtColor(v_crop, cv2.COLOR_BGR2GRAY)
        h_blur = cv2.GaussianBlur(h_gray, (5, 5), 0)
        _, h_binary = cv2.threshold(h_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        h_ranges = detect_panel_ranges(h_binary, axis='horizontal', threshold=threshold)

        for x1, x2 in h_ranges:
            sub_panel = v_crop[:, x1:x2]

            # 여백 제거
            panel_gray = cv2.cvtColor(sub_panel, cv2.COLOR_BGR2GRAY)
            _, panel_mask = cv2.threshold(panel_gray, 240, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(panel_mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                sub_panel = sub_panel[y:y + h, x:x + w]

            if sub_panel.size == 0 or sub_panel.shape[0] < 20 or sub_panel.shape[1] < 20:
                continue

            # 재귀적으로 더 쪼갤 수 있는지 확인
            sub_gray = cv2.cvtColor(sub_panel, cv2.COLOR_BGR2GRAY)
            sub_blur = cv2.GaussianBlur(sub_gray, (5, 5), 0)
            _, sub_binary = cv2.threshold(sub_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            sub_v = detect_panel_ranges(sub_binary, axis='vertical', threshold=threshold)
            sub_h = detect_panel_ranges(sub_binary, axis='horizontal', threshold=threshold)

            if len(sub_v) > 1 or len(sub_h) > 1:
                # 더 쪼갤 수 있다면 재귀 호출
                added_count, panel_idx = recursively_split_and_save(sub_panel, base_name, panel_idx, save_dir, depth + 1, max_depth)
                total_saved += added_count
            else:
                # 저장
                safe_base_name = ''.join(c for c in base_name if c.isalnum() or c in ('_', '-'))
                filename = f"{safe_base_name}_panel_{panel_idx:02}.jpg"
                save_path = os.path.join(save_dir, filename)

                success = cv2.imwrite(save_path, sub_panel)
                if success:
                    panel_idx += 1
                    total_saved += 1

    return total_saved, panel_idx


def split_image_panels(input_folder="raw_images", output_folder="cut_panels/회차제목"):
    # 임시 폴더: 같은 상위 디렉토리 내에 생성
    parent_dir = os.path.dirname(output_folder)
    final_name = os.path.basename(output_folder)
    temp_output_folder = "cut_panels/temp"

    os.makedirs(temp_output_folder, exist_ok=True)
    images = sorted(os.listdir(input_folder))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] 이미지 불러오기 실패: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        inverted = cv2.bitwise_not(gray) if avg_brightness > 127 else gray.copy()
        blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        v_ranges = detect_panel_ranges(binary, axis='vertical')
        base_name = os.path.splitext(img_name)[0]
        saved_count = 0
        panel_idx = 1

        for y1, y2 in v_ranges:
            vertical_strip = img[y1:y2, :]
            added_count, panel_idx = recursively_split_and_save(vertical_strip, base_name, panel_idx, temp_output_folder)
            saved_count += added_count

        print(f"✅ {img_name}: {saved_count}컷 저장 완료")

    try:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(os.path.dirname(output_folder), exist_ok=True)
        shutil.move(temp_output_folder, output_folder)
        print(f"\n📁 폴더명을 'temp' → '{output_folder}'로 이동 완료!")
    except Exception as e:
        print(f"[!] 폴더명 변경 실패: {e}")

    print(f"\n🎉 전체 컷 분할 및 저장 완료! 저장 경로: {output_folder}")


if __name__ == "__main__":
    print(f"0. 특정 회차만 다운로드\n")
    print(f"1. 특정 회차부터 특정 회차까지 다운로드\n")
    print(f"2. 특정 페이지부터 특정 페이지까지 다운로드\n")
    menu = int(input("메뉴를 선택해 주세요 : ").strip())

    if menu == 0:
        limitHead = int(input("어떤 회차를 다운로드할까요? : ").strip())
        limitBack = limitHead
    elif menu == 1:
        limitHead = int(input("어떤 회차부터 다운로드할까요? : ").strip())
        limitBack = int(input("어떤 회차까지 다운로드할까요? : ").strip())
    elif menu == 2:
        limitHead = int(input("어떤 페이지부터 다운로드할까요? (예: 1): ").strip())
        limitBack = int(input("어떤 페이지까지 다운로드할까요? (예: 4): ").strip())
    else:
        exit()

    pagenum = 0
    pagenumlimit = 1

    if menu == 2:
        pagenum = limitHead
        limitHead = 1
        pagenumlimit = limitBack
        limitBack = 20

    # 회차 - 페이지 찾기
    if menu == 1 or menu == 0:
        while limitHead > 20:
            pagenum += 1
            limitHead -= 20

        while limitBack > 20:
            pagenumlimit += 1
            limitBack -= 20

    # limit : 현재 페이지에서 받을 회차의 수
    # pagenum = 현재 페이지
    # pagenumlimit = 마지막 페이지

    while pagenum < pagenumlimit:
        pagenum += 1

        limit = 20
        # 마지막 페이지면
        if pagenum == pagenumlimit:
            limit = limitBack

        list_url = f"https://comic.naver.com/webtoon/list?titleId=64997&page={pagenum}&sort=ASC"

        episode_urls = get_episode_urls(list_url, limit=limit)
        if not episode_urls:
            print("❌ 회차 URL을 추출하지 못했습니다.")
        else:
            print(f"🔗 {pagenum}페이지에서 {len(episode_urls)}개의 회차 URL 추출 완료.\n")

            for idx, (ep_url, ep_title) in enumerate(episode_urls):
                # 첫 페이지면 특정 회차 스킵
                if limitHead > 1:
                    limitHead -= 1
                    continue

                print(f"📥 [{ep_title}] 이미지 다운로드 중...\nURL: {ep_url}")
                raw_dir = f"raw_images/episode_{idx+1:03}"
                count = get_webtoon_images_with_selenium(ep_url, save_dir=raw_dir)

                if count > 0:
                    cut_dir = f"cut_panels/{ep_title}"
                    print(f"🧩 [{ep_title}] 컷 분할 중...")
                    split_image_panels(input_folder=raw_dir, output_folder=cut_dir)

            print(f"\n✅ {pagenum}페이지 완료!")
    print("\n✅ 전체 완료!")