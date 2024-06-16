# Commented out IPython magic to ensure Python compatibility.
import ee
from sentry_sdk.crons import monitor

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
# Здесь нужно указать имя своего проекта в gee
ee.Initialize(project='10anilaan')

# %matplotlib inline

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set()


def ee_array_to_df(arr, list_of_bands, dropna=True):
    """Формирует из клиентского списка ee.Image.getRegion объект pandas.DataFrame.

    Аргументы:
    arr -- массив, возвращаемый методом .getInfo()
    list_of_bands -- список каналов
    dropna -- если True, исключает строки с пропущенными значениями (default True)
    """
    df = pd.DataFrame(arr)

    # Добавление звголовков
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Сохранение необходимых столбцов (и удаление остальных)
    df = df[['time', *list_of_bands]]
    if dropna:
        # Удаление строк, где есть хотя бы одно пропущенное значение
        df = df.dropna()

    # Перевод данных в числовые
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Перевод времени в милисикундак в datetime
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Из времени оставляем только datetime
    df = df[['datetime', *list_of_bands]]

    return df


def applyScaleFactorsL5L7(image):
    """Применяет scale и offset к данным Landsat5 и Landsat7."""
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True).addBands(thermalBand, None, True)


def applyScaleFactorsL8L9(image):
    """Применяет scale и offset к данным Landsat8 и Landsat9."""
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)


def L7_func(image):
    """Заполняет пустые полосы Landsat7 средними (полосы появляются из-за отказа прибора SLC)."""
    filled = ee.Image(image.focalMean(1, 'square', 'pixels', 2) \
                      .copyProperties(image, image.propertyNames()))
    return filled.blend(image)


def L8L9_func(image):
    """Переименовывает каналы Landsat8 и 9 (для их соответствия L5 и L7)."""
    return image.rename(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6', 'QA_PIXEL'])


def applyScaleFactorsS2(image):
    """Применяет scale и offset к данным Sentinel2."""
    opticalBands = image.select('B.*').multiply(0.0001)
    return image.addBands(opticalBands, None, True)


def applyScaleFactorsMTemp(image):
    """Применяет scale и offset к данным MODIS Terra Land Surface Temperature."""
    thermalBands = image.select('LST_.*').multiply(0.02)
    return image.addBands(thermalBands, None, True)


# Функции маскирования облаков

def maskLandsat(image):
    """Маскирует облака в данных Landsat (кроме Landsat7)."""
    # 3 и 4 биты отвечают за облака и тени от них
    cloudsBitMask = 1 << 3
    cloudShadowBitMask = 1 << 4
    # Получаем канал с данными от CFMASK
    qa = image.select('QA_PIXEL')
    # Оба бита должны содержать 0 (чистое небо)
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
        .And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    # Возвращаем маскированое изображение
    return image.updateMask(mask) \
        .copyProperties(image, ["system:time_start"])


def maskLandsat7(df, list_of_bands, dropna=True):
    """Маскирует облака в данных Landsat7 после их выгрузки."""
    cloudsBitMask = 1 << 3
    cloudShadowBitMask = 1 << 4
    qa = df['QA_PIXEL']
    mask = (qa & cloudShadowBitMask == 0) & (qa & cloudsBitMask == 0)
    if dropna:
        return df.drop('QA_PIXEL', axis=1)[mask]
    else:
        new_df = df.copy()
        new_df.loc[:, list_of_bands][~mask] = None
        return new_df.drop('QA_PIXEL', axis=1)


def maskSentinel(image):
    """Маскирует облака в данных Sentinel."""
    # 10 и 11 биты отвечают за непрозрачные и перистые облака
    opaqueBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Получаем канал с данными от CFMASK
    qa = image.select('QA60')
    # Оба бита должны содержать 0 (чистое небо)
    mask = qa.bitwiseAnd(opaqueBitMask).eq(0) \
        .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    # Возвращаем маскированое изображение
    return image.updateMask(mask) \
        .copyProperties(image, ["system:time_start"])


def getLansatPointData(poi, res, first_date, end_date, CLOUD_MAX, mask_clouds=True, save=False):
    """Выгружает данные Landsat (c 5 по 9) по точке.

    Аргументы:
    poi -- точка интереса, объект ee.Geometry.Point
    res -- разрешение в метрах (размер стороны квадрата-пикселя)
    first_date -- начальная дата
    end_date -- конечная дата
    CLOUD_MAX -- максимальная облачность гранул в процентах
    mask_clouds -- если True, маскирует облака (default True)
    save -- если True, сохраняет данные в формате csv (default False)
    """
    L5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
        .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_MAX)) \
        .filterBounds(poi) \
        .filterDate(first_date, end_date) \
        .select(['SR_B.*', 'ST_B.*', 'QA_PIXEL']) \
        .map(applyScaleFactorsL5L7)

    L7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
        .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_MAX)) \
        .filterBounds(poi) \
        .filterDate(first_date, end_date) \
        .select(['SR_B.*', 'ST_B.*', 'QA_PIXEL']) \
        .map(applyScaleFactorsL5L7) \
        .map(L7_func)

    L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_MAX)) \
        .filterBounds(poi) \
        .filterDate(first_date, end_date) \
        .select(['SR_B[2-7]', 'ST_B10', 'QA_PIXEL']) \
        .map(applyScaleFactorsL8L9) \
        .map(L8L9_func)

    L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
        .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_MAX)) \
        .filterBounds(poi) \
        .filterDate(first_date, end_date) \
        .select(['SR_B[2-7]', 'ST_B10', 'QA_PIXEL']) \
        .map(applyScaleFactorsL8L9) \
        .map(L8L9_func)

    L_merged = L5.merge(L8.merge(L9))

    if mask_clouds:
        L_merged = L_merged.map(maskLandsat)

    L_poi = L_merged \
        .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4']) \
        .getRegion(poi, res) \
        .getInfo()

    L_df = ee_array_to_df(L_poi, ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4'])

    L7_poi = L7 \
        .select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'QA_PIXEL']) \
        .getRegion(poi, res)
    L7_poi = L7_poi.getInfo()

    L7_df = ee_array_to_df(L7_poi, ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'QA_PIXEL']).astype({'QA_PIXEL': 'int'})

    if mask_clouds:
        L7_df = maskLandsat7(L7_df_cloudy, ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'QA_PIXEL'])
    else:
        L7_df = L7_df.drop('QA_PIXEL', axis=1)

    L_df = pd.concat([L_df, L7_df])
    L_df.sort_values(by=['datetime'], ignore_index=True, inplace=True)
    L_df['date'] = L_df['datetime'].dt.date
    L_df = L_df[['date', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4']]

    L_df = L_df.eval(
        '''
        NDVI = (SR_B4 - SR_B3) / (SR_B4 + SR_B3)
        GNDVI = (SR_B4 - SR_B2) / (SR_B4 + SR_B2)
        DVI = SR_B4 - SR_B3
        OSAVI = 1.16 * (SR_B4 - SR_B3) / (SR_B4 + SR_B3 + 0.16)
        ExG = (2 * SR_B2 - SR_B3 - SR_B1) / (SR_B1 + SR_B2 + SR_B3)
        ExR = (1.4 * SR_B3 - SR_B2) / (SR_B1 + SR_B2 + SR_B3)
        ExG_subtract_ExR = ExG - ExR
        NDI = (SR_B2 - SR_B3) / (SR_B2 + SR_B3)
        '''
    )

    L_df = L_df.drop(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4'], axis=1)
    L_df = L_df.groupby('date').max()

    if save:
        L_df.to_csv('Landsat.csv')

    return L_df


def getSentinelPointData(poi, res, first_date, end_date, CLOUD_MAX, mask_clouds=True, save=False):
    """Выгружает данные Sentinel2 по точке.

    Аргументы:
    poi -- точка интереса, объект ee.Geometry.Point
    res -- разрешение в метрах (размер стороны квадрата-пикселя)
    first_date -- начальная дата
    end_date -- конечная дата
    CLOUD_MAX -- максимальная облачность гранул в процентах
    mask_clouds -- если True, маскирует облака (default True)
    save -- если True, сохраняет данные в формате csv (default False)
    """
    S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_MAX)) \
        .filterBounds(poi) \
        .filterDate(first_date, end_date) \
        .select(['B.*', 'QA60']) \
        .map(applyScaleFactorsS2)

    if mask_clouds:
        S2 = S2.map(maskSentinel)

    S_poi = S2 \
        .select(['B2', 'B3', 'B4', 'B6', 'B8']) \
        .getRegion(poi, res) \
        .getInfo()

    S_df = ee_array_to_df(S_poi, ['B2', 'B3', 'B4', 'B6', 'B8'])
    S_df['date'] = S_df['datetime'].dt.date
    S_df = S_df[['date', 'B2', 'B3', 'B4', 'B6', 'B8']]

    S_df = S_df.eval(
        '''
        NDVI = (B8 - B4) / (B8 + B4)
        GNDVI = (B8 - B3) / (B8 + B3)
        DVI = B8 - B4
        OSAVI = 1.16 * (B8 - B4) / (B8 + B4 + 0.16)
        ExG = (2 * B3 - B4 - B2) / (B2 + B3 + B4)
        ExR = (1.4 * B4 - B3) / (B2 + B3 + B4)
        ExG_subtract_ExR = ExG - ExR
        NDI = (B3 - B4) / (B3 + B4)
        NDRE  = (B8 - B6) / (B8 + B6)
        '''
    )

    S_df = S_df.drop(['B2', 'B3', 'B4', 'B6', 'B8'], axis=1)
    S_df = S_df.groupby('date').max()

    if save:
        S_df.to_csv('Sentinel.csv')

    return S_df


def getMODISCombinedPointData(poi, res, first_date, end_date, save=False):
    """Выгружает данные NDVI MODIS (ежедневные 16-идневные композиты) по точке.

    Аргументы:
    poi -- точка интереса, объект ee.Geometry.Point
    res -- разрешение в метрах (размер стороны квадрата-пикселя)
    first_date -- начальная дата
    end_date -- конечная дата
    save -- если True, сохраняет данные в формате csv (default False)
    """
    MODIS = ee.ImageCollection("MODIS/MOD09GA_006_NDVI") \
        .filterBounds(poi) \
        .filterDate(first_date, end_date) \
        .select(['NDVI']) \
        .getRegion(poi, res)

    M_poi = MODIS.getInfo()

    M_df = ee_array_to_df(M_poi, ['NDVI'])
    M_df['date'] = M_df['datetime'].dt.date
    M_df = M_df[['date', 'NDVI']]
    M_df = M_df.groupby('date').max()

    if save:
        M_df.to_csv('MODIS_combined_NDVI.csv')

    return M_df


def getMODISTempPointData(poi, res, first_date, end_date, save=False):
    """Выгружает данные MODIS Terra Land Surface Temperature по точке.

    Аргументы:
    poi -- точка интереса, объект ee.Geometry.Point
    res -- разрешение в метрах (размер стороны квадрата-пикселя)
    first_date -- начальная дата
    end_date -- конечная дата
    save -- если True, сохраняет данные в формате csv (default False)
    """
    MODIS = ee.ImageCollection("MODIS/061/MOD11A2") \
        .filterBounds(poi) \
        .filterDate(first_date, end_date) \
        .select(['LST_Day_1km']) \
        .map(applyScaleFactorsMTemp) \
        .getRegion(poi, res)

    M_poi = MODIS.getInfo()

    M_df = ee_array_to_df(M_poi, ['LST_Day_1km'])
    M_df['date'] = M_df['datetime'].dt.date
    M_df = M_df[['date', 'LST_Day_1km']]
    M_df = M_df.groupby('date').max()

    if save:
        M_df.to_csv('MODIS_Land_Surface_Temperature.csv')

    return M_df


def medvedeva_smoothing(index_series, n_dots, n_rounds):
    index = index_series.copy()
    for i in range(n_rounds):
        for i in range(1, index.shape[0] - n_dots + 2):
            if index.iloc[i] < index.iloc[i - 1] and index.iloc[i] < index.iloc[i + 1]:
                continue

            if i < index.shape[0] - n_dots:
                r_border = i + n_dots - 1
            else:
                r_border = index.shape[0] - 2

            while not ((index.iloc[i + 1:r_border] < index.iloc[i]).all() and
                       (index.iloc[i + 1:r_border] < index.iloc[r_border]).all() and
                       (index.iloc[r_border] >= index.iloc[r_border - 1] or index.iloc[r_border] >= index.iloc[
                           r_border + 1])):
                r_border -= 1
                if r_border == i + 1:
                    break

            if r_border == i + 1:
                continue

            index.iloc[i:r_border + 1] = np.linspace(index.iloc[i], index.iloc[r_border], r_border - i + 1)
    return index


def drawData(df, date_offset=True, legend=False, figsize=(10, 6), save=False):
    """Строит графики по выгруженным данным.

    Аргументы:
    df -- pandas.DataFrame, полученный от любой из функций выгрузки данных по точке
    date_offser -- если True, метка с годом N будет означать конец N-ого года (defaul True)
    legend -- если True, то легенда отображается (default False)
    figsize -- пользовательский размер графика, аргумент для plt.subplots (default (10, 6))
    save -- сохранить графики в формате png
    """
    for ind in df.columns:
        fig, ax = plt.subplots(figsize=figsize)

        if date_offset:
            ax.plot(df.index - pd.offsets.DateOffset(years=1), df[ind])
        else:
            ax.plot(df.index, df[ind])

        if legend:
            ax.set_title(ind)

        ax.xaxis.set_tick_params(reset=True)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        if save:
            fig.savefig(ind + '.png', dpi=1200)


@monitor(monitor_slug='monitoring-tree-daily')
def run():
    # Задаем область интереса
    roi = ee.Geometry.Polygon([[[119.843668, 65.54141],
                                [119.843668, 65.532524],
                                [119.865987, 65.532524],
                                [119.865987, 65.54141],
                                [119.843668, 65.54141]]])
    # Задаем масимальную облачность гранул
    CLOUD_MAX = 100
    # Задаем разрешение в метрах
    res = 30
    # Задаем начало и конец временного периода
    first_date = '2022-01-01'
    end_date = '2023-01-01'

    # Средняя периодичность получения снимков в днях (ставится полурандомно)
    freq = 2
    # Количество выгружаемых слоев
    bands = 2
    # Расчет количества пикселей в пределах полигона и предела количества
    # выгружаемых месяцев (число 1048576 взято из документации)
    pixel_count = roi.area().getInfo() / (res ** 2)
    image_count = int(1048576 / (pixel_count * bands))
    months_delta = int(freq * image_count / 31)
    print(f"Макимум месяцев: {months_delta}\nМаксимум лет: {months_delta // 12}")

    # Фильтруем коллекции Landsat по максимальной облачности гранул, полигону,
    # дате, выбираем все полезные каналы и применяем все функции

    L5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
        .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_MAX)) \
        .filterBounds(roi) \
        .filterDate(first_date, end_date) \
        .select(['SR_B.*', 'ST_B.*', 'QA_PIXEL']) \
        .map(applyScaleFactorsL5L7)

    L7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
        .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_MAX)) \
        .filterBounds(roi) \
        .filterDate(first_date, end_date) \
        .select(['SR_B.*', 'ST_B.*', 'QA_PIXEL']) \
        .map(applyScaleFactorsL5L7) \
        .map(L7_func)

    L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_MAX)) \
        .filterBounds(roi) \
        .filterDate(first_date, end_date) \
        .select(['SR_B[2-7]', 'ST_B10', 'QA_PIXEL']) \
        .map(applyScaleFactorsL8L9) \
        .map(L8L9_func)

    L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
        .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_MAX)) \
        .filterBounds(roi) \
        .filterDate(first_date, end_date) \
        .select(['SR_B[2-7]', 'ST_B10', 'QA_PIXEL']) \
        .map(applyScaleFactorsL8L9) \
        .map(L8L9_func)

    # Фильтруем коллекцию Sentinel

    S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_MAX)) \
        .filterBounds(roi) \
        .filterDate(first_date, end_date) \
        .select(['B.*', 'QA60']) \
        .map(applyScaleFactorsS2)

    # При необходимости создаем объединение коллекций Landsat
    L_merged = L5.merge(L8.merge(L9))

    # Вырезаем облака, выбираем нужные каналы и выгружаем список по области
    # (на стороне сервера)
    L_roi = L_merged \
        .map(maskLandsat) \
        .select(['SR_B3', 'SR_B4']) \
        .getRegion(roi, res)

    # Запрашиваем список с сервера
    L_roi = L_roi.getInfo()

    # Преобразуем список в pd.DataFrame
    L_df = ee_array_to_df(L_roi, ['longitude', 'latitude', 'SR_B3', 'SR_B4'], dropna=False)

    # Из-за особенностей фильтрации Landsat 7 выгружаем отдельно
    # Выбирать QA_PIXEL - обязательно!
    L7_roi = L7 \
        .select(['SR_B3', 'SR_B4', 'QA_PIXEL']) \
        .getRegion(roi, res)

    L7_roi = L7_roi.getInfo()

    L7_df_cloudy = ee_array_to_df(L7_roi, ['longitude', 'latitude', 'SR_B3', 'SR_B4', 'QA_PIXEL'])
    L7_df_cloudy = L7_df_cloudy.astype({'QA_PIXEL': 'int'})

    L7_df = maskLandsat7(L7_df_cloudy, ['SR_B3', 'SR_B4', 'QA_PIXEL'], dropna=False)

    L_df = pd.concat([L_df, L7_df])
    L_df.sort_values(by=['datetime'], ignore_index=True, inplace=True)

    S_roi = S2 \
        .map(maskSentinel) \
        .select(['B4', 'B8']) \
        .getRegion(roi, res)

    S_roi = S_roi.getInfo()

    S_df = ee_array_to_df(S_roi, ['longitude', 'latitude', 'B4', 'B8'], dropna=False)
