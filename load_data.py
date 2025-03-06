from fastapi import HTTPException
import mysql.connector
import os
from dotenv import load_dotenv
import boto3
import datetime as dt
from photo_features import *
import logging

import warnings

warnings.filterwarnings("ignore")


# S3 Keys
bucket_name_dict = {
    "company_name1": "prod-company_name1.ru",
    "company_name2": "prod-company_name2.ru"
}
logger = logging.getLogger(__name__)
load_dotenv()

endpoint_url = "https://s3.ru-1.bucket.companycloud.ru"
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
region_name = 'ru-1'
prefix = "uploads/"
download_folder = "photo_examples/"  # Folder where photos will be saved


def convert_data(data):
    """
    Convert data types for compatibility with FastAPI
    """
    if isinstance(data, dict):
        return {key: convert_data(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return tuple(convert_data(value) for value in data)
    elif isinstance(data, list):
        return [convert_data(value) for value in data]
    elif isinstance(data, IFDRational):
        if hasattr(data, "denominator") and data.denominator != 0:
            return float(data)
        else:
            return None
    elif type(data) in [np.float16, np.float32, np.float64]:
        return float(data)
    elif type(data) in [np.int16, np.int32, np.int64, np.uint8]:
        return int(data)
    else:
        return data


@time_it
def mysql_query(company: str, sql: str) -> pd.DataFrame:
    if company.lower() in ["COMPANY_NAME1".lower(), "CN1".lower()]:
        host_id = "0.0.0.17"
        database_id = "prod_company_name1_api"
    if company.lower() in ["COMPANY_NAME2".lower(), "CN2".lower()]:
        host_id = "0.0.0.37"
        database_id = "prod_company_name2_api"
    conn = mysql.connector.connect(
        host=host_id,
        database=database_id,
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


@time_it
def load_hashedFilename(company_name, loan_id):

    sql_text = f"""
    select 
        loanid as loan_id,
        file.id as file_id, file.userId as user_id, 
        file.name, file.insertDate, file.hashedFilename, file.validation_status
    from (
        select 
            id as loanid, userId
        from loan as l
        where l.id = {loan_id}
    ) as l 
    left join file on file.userId=l.userId and `type` = 'user'
    where hashedFilename is not null
    """
    try:
        df = mysql_query(company_name, sql_text)
        logger.info(f"Download dataframe, shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise HTTPException(status_code=500, detail="Database connection issue")


@time_it
def download_file_from_s3(
    company_name,
    object_key,
    download_path,
    endpoint_url,
    aws_access_key,
    aws_secret_key,
    region_name="ru-1",
):

    bucket_name = bucket_name_dict[company_name]
    # Create a session and S3 client with a custom endpoint for Selectel
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name,
    )
    s3 = session.client("s3", endpoint_url=endpoint_url)

    try:
        # Download the file from S3
        s3.download_file(bucket_name, object_key, download_path)
        logger.info(f"File {object_key} successfully downloaded to {download_path}")
        return download_path

    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise HTTPException(status_code=500, detail="S3 connection issue")


@time_it
def extract_photo_features(company_name, df):

    dict_photo_features = {}
    uploads_list = [prefix + file_name for file_name in df.hashedFilename]  # [:10]

    for object_key in uploads_list:

        logger.info("Start processing photos")
        start_time = dt.datetime.now()

        mask_index = df.hashedFilename == object_key.replace(prefix, "")
        df_tmp = df[mask_index].reset_index()

        download_path = object_key.replace(
            prefix, ""
        )  # Path to download the file locally

        file_id = df_tmp["file_id"][0]

        dict_photo_features[file_id] = {}
        dict_photo_features[file_id]["file_name"] = df_tmp["name"][0]
        dict_photo_features[file_id]["hashedFilename"] = df_tmp["hashedFilename"][0]

        # Step 1: Download the file from S3
        start_time_locale = dt.datetime.now()
        downloaded_image_path = download_file_from_s3(
            company_name,
            object_key,
            download_path,
            endpoint_url,
            aws_access_key,
            aws_secret_key,
        )
        logger.info(
            f"Time used for Downloading = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n"
        )

        start_time_locale = dt.datetime.now()
        # Step 2: Extract EXIF data from the downloaded file
        if downloaded_image_path:

            # image = Image.open(downloaded_image_path)
            # display(image)

            exif = extract_exif(downloaded_image_path)

            cnt_exif = 0

            # Output the result
            if exif:
                # print("EXIF data:")
                dict_photo_features[file_id]["exif"] = {}

                for tag, value in exif.items():
                    cnt_exif += 1
                    if type(value) == str:
                        value = value.replace("\x00", "")[:128]
                    dict_photo_features[file_id]["exif"][tag] = value
                    if type(value) == bytes:
                        dict_photo_features[file_id]["exif"][tag] = "bytes"

                dict_photo_features[file_id]["upl_to_last_change_photo_diff"] = (
                    calc_timediff(
                        df_tmp["insertDate"][0],
                        dict_photo_features[file_id]["exif"].get("DateTime"),
                    )
                )
                dict_photo_features[file_id]["upl_to_original_photo_diff"] = (
                    calc_timediff(
                        df_tmp["insertDate"][0],
                        dict_photo_features[file_id]["exif"].get("DateTimeOriginal"),
                    )
                )
                dict_photo_features[file_id]["upl_to_digitized_photo_diff"] = (
                    calc_timediff(
                        df_tmp["insertDate"][0],
                        dict_photo_features[file_id]["exif"].get("DateTimeDigitized"),
                    )
                )
                dict_photo_features[file_id]["last_change_photo_to_original_diff"] = (
                    calc_timediff(
                        dict_photo_features[file_id]["exif"].get("DateTime"),
                        dict_photo_features[file_id]["exif"].get("DateTimeOriginal")
                    )
                )

            dict_photo_features[file_id]["cnt_exif"] = cnt_exif

            # dict_photo_features[file_id]['ts'] = type(df_tmp['insertDate'])
            image_path = downloaded_image_path

            start_time_locale = dt.datetime.now()
            sharpness = calculate_sharpness(image_path)
            dict_photo_features[file_id]["sharpness"] = sharpness
            # print(f"Time used for sharpness = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            start_time_locale = dt.datetime.now()
            brightness = calculate_brightness(image_path)
            dict_photo_features[file_id]["brightness"] = brightness
            # print(f"Time used for brightness = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            start_time_locale = dt.datetime.now()
            for threshold in [235, 230, 225]:
                overexposure = check_overexposure(image_path, threshold=threshold)
                dict_photo_features[file_id][f"overexposure_{threshold}"] = overexposure
            # print(f"Time used for overexposure_ = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            start_time_locale = dt.datetime.now()
            for threshold in [20, 25, 30]:
                darkness = check_darkness(image_path, threshold=threshold)
                dict_photo_features[file_id][f"darkness_{threshold}"] = darkness
            # print(f"Time used for darkness_ = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            start_time_locale = dt.datetime.now()
            contrast = calculate_contrast(image_path)
            dict_photo_features[file_id]["contrast"] = contrast
            # print(f"Time used for contrast = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            start_time_locale = dt.datetime.now()
            colorfulness = calculate_colorfulness(image_path)
            dict_photo_features[file_id]["colorfulness"] = colorfulness
            # print(f"Time used for colorfulness = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            # # Histogram
            # hist = color_histogram(image_path)
            # print("Color histogram created.")

            # # Texture features
            # texture = texture_features(image_path)
            # print("Texture features:", texture)

            # Object count
            start_time_locale = dt.datetime.now()
            objects_count = count_objects(image_path)
            dict_photo_features[file_id]["objects_count"] = objects_count
            # print(f"Time used for objects_count = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            # Aspect ratio
            start_time_locale = dt.datetime.now()
            aspect = aspect_ratio(image_path)
            dict_photo_features[file_id]["aspect_ratio"] = aspect
            # print(f"Time used for aspect_ratio = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            # Object density
            start_time_locale = dt.datetime.now()
            density = object_density(image_path)
            dict_photo_features[file_id]["density"] = density
            # print(f"Time used for density = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            # print(f"Object density: {density:.2f}%")

            # Dynamic range
            # dyn_range = dynamic_range(image_path)
            # print(f"Dynamic range: {dyn_range}")

            # # FFT analysis
            # fft_result = fft_analysis(image_path)
            # print("FFT analysis completed.")

            # # ELA
            # ela_result = error_level_analysis(image_path)
            # cv2.imwrite("ela_result.jpg", ela_result)
            # print("ELA analysis completed. Result saved as 'ela_result.jpg'.")

            # Face detection
            start_time_locale = dt.datetime.now()
            faces = detect_face(image_path)
            dict_photo_features[file_id]["cntfaces"] = len(faces)
            # print(f"Time used for faces = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            # if len(faces) > 0:
            #     print(f"Detected {len(faces)} face(s).")
            # else:
            #     print("No faces detected.")

            # # Shadows and highlights
            # shadows_highlights = analyze_shadows_and_highlights(image_path)
            # print("Shadows and highlights:", shadows_highlights)

            start_time_locale = dt.datetime.now()
            faces_density = face_density(image_path, min_detection_confidence=0.5)
            dict_photo_features[file_id]["faces_density"] = faces_density
            # print(f"Time used for faces_density = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            #             start_time_locale = dt.datetime.now()
            #             if file_type in ['passport','registration','photo']:
            #                 text, text_number = extract_text(image_path , lang='rus+eng')
            #             dict_photo_features[file_id]['file_text'] = file_text
            #             dict_photo_features[file_id]['file_text_numbers'] = file_text_numbers
            #             print(f"Time used for file_text = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            #             start_time_locale = dt.datetime.now()
            #             if ('card' in file_type) | (file_type == 'passport'):
            #                 file_numbers_text = extract_card_number_q(image_path)
            #                 dict_photo_features[file_id]['file_numbers_text_q'] = file_numbers_text
            #             print(f"Time used for file_numbers_text_q = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n")

            # Delete the downloaded file after processing (optional)
            start_time_locale = dt.datetime.now()
            if os.path.exists(downloaded_image_path):
                os.remove(downloaded_image_path)
                logger.info(f"File {downloaded_image_path} deleted after processing.")
            logger.info(
                f"Time used for Delete = {round((dt.datetime.now()-start_time_locale).total_seconds(), 1)} seconds \n"
            )

            logger.info("Finish")
            logger.info(
                f"Time used = {round((dt.datetime.now()-start_time).total_seconds(), 1)} seconds \n"
            )
            #dict_photo_features[file_id]["insertDate"] = df_tmp["insertDate"][0]
            dict_photo_features = calc_unique_doctypes(dict_photo_features)
            dict_photo_features = {
                int(key): value for key, value in dict_photo_features.items()
            }

    # seconds_used = round((dt.datetime.now()-start_time_all).total_seconds(), 1)
    return dict_photo_features
