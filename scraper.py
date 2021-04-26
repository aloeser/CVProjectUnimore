import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup


def make_dataset(dir_to_save, country, url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    headers['Cookie'] = response.headers['Set-Cookie']
    soup = BeautifulSoup(response.text, 'html.parser')
    table_cells = soup.find_all("a", attrs={'class': 'cell'})
    country_dir = dir_to_save + country + '/'
    os.makedirs(country_dir)
    print(f'Downloading images for country: {country}')
    for n, cell in enumerate(tqdm(table_cells)):
        for k in ('data-tooltip-code', 'data-tooltip-sample', 'data-tooltip-code'):
            if k not in cell.attrs.keys():
                continue
        label = cell.attrs['data-tooltip-code']
        if int(label.split('-')[-1]) > 2017:
            continue
        if 'cent' in label:
            coin = label.split('-')[1] + 'ct'
        else:
            coin = label.split('-')[1]
        out_dir = country_dir + coin + '/'
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir) 
        for i in (1, 2):
            img_path = cell.attrs['data-tooltip-imgpath']                           \
                        + '/'                                                       \
                        + cell.attrs['data-tooltip-sample']                         \
                        + '-' + str(i) + 's/'                                       \
                        + cell.attrs['data-tooltip-code']                           \
                        + '.jpg'
            response = requests.get(img_path, headers=headers)
            with open(out_dir + label + ('-back' if i==1 else '-front') + '.jpg', "wb") as file:
                file.write(response.content)


if __name__ == '__main__':
    country_dict = {
        'Italy': 'https://en.ucoin.net/table/?country=italy&period=314',
        'Germany': 'https://it.ucoin.net/table/?country=germany&period=1'
    }
    dir_to_save = './data'
    if dir_to_save[-1] != '/':
        dir_to_save += '/'
    if os.path.isdir(dir_to_save):
        print('Directory already exists! Exiting...')
        exit()
    elif not os.path.isdir(dir_to_save):
        os.makedirs(dir_to_save)
    for country, url in country_dict.items():
        make_dataset(dir_to_save, country, url)
