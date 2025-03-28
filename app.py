# # app.py
# from flask import Flask, render_template, request, redirect, url_for, flash,session
# import requests
# import json
# import base64
# import cv2
# import numpy as np
# import os
# import re
# from dotenv import load_dotenv
#
# load_dotenv()
#
# app_secret_key = os.getenv("SECRET_KEY")
# gemini_api_key = os.getenv("GEMINI_API_KEY")
#
# app = Flask(__name__)
# app.secret_key = app_secret_key  # Replace with a real secret key
#
# # Load voter information
# with open("voter_data.json", "r") as file:
#     VOTER_DATA = json.load(file)
#
# # Gemini API configuration (replace with your actual API key)
# GEMINI_API_KEY = gemini_api_key
#
#
# def get_text_from_image(image_bytes, mime_type):
#     """Extract structured details from the Voter ID card using Gemini 1.5 API."""
#     url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
#
#     headers = {"Content-Type": "application/json"}
#
#     prompt = """
#     Extract structured details from the voter card image.
#     The output should be a valid JSON with fields:
#     {
#         "Name": "",
#         "Age": "",
#         "Voter ID": "",
#         "Father's Name": "",
#         "Gender": "",
#         "Address": "",
#         "Date Of Birth" : ""
#     }
#     If any field is not found, return an empty string for it.
#     """
#
#     # Convert image to base64
#     image_base64 = base64.b64encode(image_bytes).decode("utf-8")
#
#     payload = {
#         "contents": [
#             {
#                 "parts": [
#                     {"text": prompt},
#                     {"inline_data": {"mime_type": mime_type, "data": image_base64}}
#                 ]
#             }
#         ]
#     }
#
#     response = requests.post(url, headers=headers, data=json.dumps(payload))
#
#     if response.status_code == 200:
#         result = response.json()
#         return result
#     else:
#         return {"error": f"API Error {response.status_code}: {response.text}"}
#
#
# # def match_fingerprint(sample_fingerprint, voter_id):
# #     """
# #     Perform fingerprint matching similar to the Streamlit implementation
# #     """
# #     # Initialize SIFT
# #     sift = cv2.SIFT_create(nfeatures=500)  # Restrict features to save memory
# #
# #     # Detect and compute features for the sample image
# #     keypoints_1, descriptors_1 = sift.detectAndCompute(sample_fingerprint, None)
# #     if descriptors_1 is None:
# #         return False, None
# #
# #     # Folder containing real fingerprint images
# #     real_folder = "SOCOFing/Real"
# #
# #     # Best match tracking
# #     best_score = 0
# #     best_filename = None
# #
# #     # Process images
# #     for file in os.listdir(real_folder):
# #         file_path = os.path.join(real_folder, file)
# #         fingerprint_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
# #
# #         if fingerprint_image is None:
# #             print(f"Warning: Could not read image {file_path}")
# #             continue  # Skip this file
# #
# #         keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
# #         if descriptors_2 is None:
# #             print(f"Warning: No descriptors found for {file_path}")
# #             continue
# #
# #         # Match descriptors using FLANN matcher
# #         matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {})
# #         matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
# #
# #         # Apply ratio test
# #         match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]
# #
# #         # Calculate match score
# #         keypoints = min(len(keypoints_1), len(keypoints_2))
# #         match_score = (len(match_points) / keypoints) * 100 if keypoints > 0 else 0
# #
# #         # Update best match
# #         if match_score > best_score:
# #             best_score = match_score
# #             best_filename = file
# #
# #     # Check if the matched fingerprint corresponds to the voter ID
# #     if best_filename:
# #         match = re.match(r"(\d+)__", best_filename)
# #         if match:
# #             number = match.group(1)
# #             data = VOTER_DATA[int(number) - 1]
# #             return data['vid'] == voter_id, best_score
# #
# #     return False, None
#
# def match_fingerprint(sample_fingerprint, voter_id):
#     """
#     Perform fingerprint matching only for the relevant voter's stored fingerprint images.
#     """
#     # Get the index of the voter from JSON
#     voter_index = None
#     for i, voter in enumerate(VOTER_DATA):
#         if voter["vid"] == voter_id:
#             voter_index = i + 1  # Since file names start from 1
#             break
#     # print(voter_index)
#
#     if voter_index is None:
#         return False, None  # Voter ID not found
#
#     # Initialize SIFT
#     sift = cv2.SIFT_create(nfeatures=500)
#
#     # Detect and compute features for the sample fingerprint
#     keypoints_1, descriptors_1 = sift.detectAndCompute(sample_fingerprint, None)
#     if descriptors_1 is None:
#         return False, None
#
#     # Define the fingerprint folder
#     real_folder = "SOCOFing/Real"
#
#     # Best match tracking
#     best_score = 0
#     best_filename = None
#
#     # Check only the 10 fingerprint files corresponding to the voter's index
#     for i in range(1, 11):  # Assuming 10 images per voter
#         file_name_pattern = f"{voter_index}__"
#         matching_files = [file for file in os.listdir(real_folder) if file.startswith(file_name_pattern)]
#         # print(matching_files)
#
#         for file in matching_files:
#             file_path = os.path.join(real_folder, file)
#             fingerprint_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#
#             if fingerprint_image is None:
#                 continue  # Skip unreadable files
#
#             keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
#             if descriptors_2 is None:
#                 continue  # Skip images with no descriptors
#
#             # Match descriptors using FLANN matcher
#             matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {})
#             matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
#
#             # Apply ratio test
#             match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]
#
#             # Calculate match score
#             keypoints = min(len(keypoints_1), len(keypoints_2))
#             match_score = (len(match_points) / keypoints) * 100 if keypoints > 0 else 0
#
#             # Update best match
#             if match_score > best_score:
#                 best_score = match_score
#                 best_filename = file
#
#     return best_score > 50, best_score  # Assuming a match threshold of 50%
#
#
#
# @app.route('/', methods=['GET', 'POST'])
# def upload_voter_card():
#     if request.method == 'POST':
#         # Check if a file was uploaded
#         if 'voter_card' not in request.files:
#             flash('No file uploaded', 'error')
#             return redirect(request.url)
#
#         file = request.files['voter_card']
#
#         if file.filename == '':
#             flash('No selected file', 'error')
#             return redirect(request.url)
#
#         # Read image bytes
#         image_bytes = file.read()
#         mime_type = f"image/{file.filename.split('.')[-1]}"
#
#         # Extract details using Gemini API
#         extracted_data = get_text_from_image(image_bytes, mime_type)
#
#         if "candidates" in extracted_data and extracted_data["candidates"]:
#             extracted_text = extracted_data["candidates"][0]["content"]["parts"][0]["text"]
#
#             # Parse the extracted text into a dictionary
#             details = {}
#             for line in extracted_text.split("\n"):
#                 if ":" in line:
#                     key, value = line.split(":", 1)
#                     key = key.strip().replace('"', '')
#                     value = value.strip().replace('"', '').replace(',', '')
#                     details[key] = value
#
#             # Store details in session for next steps
#             session['voter_details'] = details
#             return redirect(url_for('enter_voter_id'))
#         else:
#             flash('Could not extract details from the voter card', 'error')
#
#     return render_template('upload_voter_card.html')
#
#
# @app.route('/enter-voter-id', methods=['GET', 'POST'])
# def enter_voter_id():
#     if 'voter_details' not in session:
#         flash('Please upload voter card first', 'error')
#         return redirect(url_for('upload_voter_card'))
#
#     if request.method == 'POST':
#         vid = request.form.get('voter_id', '').strip()
#         voter_details = session['voter_details']
#
#         # Verify details
#         for voter in VOTER_DATA:
#             if (voter['vid'] == vid and
#                     voter['name'].lower() == voter_details.get('Name', '').strip().lower() and
#                     voter['dob'] == voter_details.get('Date Of Birth', '').strip()):
#                 # Store verified voter ID in session
#                 session['verified_vid'] = vid
#                 return redirect(url_for('upload_fingerprint'))
#
#         flash('Voter ID verification failed', 'error')
#
#     return render_template('enter_voter_id.html')
#
#
# @app.route('/upload-fingerprint', methods=['GET', 'POST'])
# def upload_fingerprint():
#     if 'verified_vid' not in session:
#         flash('Please verify voter ID first', 'error')
#         return redirect(url_for('enter_voter_id'))
#
#     if request.method == 'POST':
#         # Check if a file was uploaded
#         if 'fingerprint' not in request.files:
#             flash('No file uploaded', 'error')
#             return redirect(request.url)
#
#         file = request.files['fingerprint']
#
#         if file.filename == '':
#             flash('No selected file', 'error')
#             return redirect(request.url)
#
#         # Read fingerprint image
#         file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
#         sample = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
#
#         # Get verified voter ID from session
#         voter_id = session['verified_vid']
#
#         # Perform fingerprint matching
#         is_match, match_score = match_fingerprint(sample, voter_id)
#
#         if is_match:
#             flash(f'Successfully Verified! Match Score: {match_score:.2f}%', 'success')
#             return redirect(url_for('verification_success'))
#         else:
#             flash('Fingerprint verification failed', 'error')
#
#     return render_template('upload_fingerprint.html')
#
#
# @app.route('/verification-success')
# def verification_success():
#     if 'verified_vid' not in session:
#         return redirect(url_for('upload_voter_card'))
#     return render_template('verification_success.html')
#
#
# if __name__ == '__main__':
#     app.run(debug=True)



import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
import os
import json
import requests  # Import requests for API calls
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
app.secret_key = os.getenv("SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load voter information
with open("voter_data.json", "r") as file:
    VOTER_DATA = json.load(file)

details = {}


# Move this function above the API routes
def get_text_from_image(image_bytes, mime_type):
    """Extract structured details from the Voter ID card using Gemini 1.5 API."""
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {"Content-Type": "application/json"}

    prompt = """
    Extract structured details from the voter card image.
    The output should be a valid JSON with fields:
    {
        "Name": "",
        "Age": "",
        "Voter ID": "",
        "Father's Name": "",
        "Gender": "",
        "Address": "",
        "Date Of Birth": ""
    }
    If any field is not found, return an empty string for it.
    """

    # Convert image to base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": image_base64}}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return {"error": f"API Error {response.status_code}: {response.text}"}


def match_fingerprint(sample_fingerprint, voter_id):
    """
    Perform fingerprint matching only for the relevant voter's stored fingerprint images.
    """
    # Get the index of the voter from JSON
    voter_index = None
    for i, voter in enumerate(VOTER_DATA):
        if voter["vid"] == voter_id:
            voter_index = i + 1  # Since file names start from 1
            break
    # print(voter_index)

    if voter_index is None:
        return False, None  # Voter ID not found

    # Initialize SIFT
    sift = cv2.SIFT_create(nfeatures=500)

    # Detect and compute features for the sample fingerprint
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample_fingerprint, None)
    if descriptors_1 is None:
        return False, None

    # Define the fingerprint folder
    real_folder = "SOCOFing/Real"

    # Best match tracking
    best_score = 0
    best_filename = None

    # Check only the 10 fingerprint files corresponding to the voter's index
    for i in range(1, 11):  # Assuming 10 images per voter
        file_name_pattern = f"{voter_index}__"
        matching_files = [file for file in os.listdir(real_folder) if file.startswith(file_name_pattern)]
        # print(matching_files)

        for file in matching_files:
            file_path = os.path.join(real_folder, file)
            fingerprint_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if fingerprint_image is None:
                continue  # Skip unreadable files

            keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)
            if descriptors_2 is None:
                continue  # Skip images with no descriptors

            # Match descriptors using FLANN matcher
            matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {})
            matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)

            # Apply ratio test
            match_points = [p for p, q in matches if p.distance < 0.1 * q.distance]

            # Calculate match score
            keypoints = min(len(keypoints_1), len(keypoints_2))
            match_score = (len(match_points) / keypoints) * 100 if keypoints > 0 else 0

            # Update best match
            if match_score > best_score:
                best_score = match_score
                best_filename = file

    return best_score > 50, best_score  # Assuming a match threshold of 50%



@app.route('/', methods=['POST'])
def upload_voter_card():
    try:
        # Check if a file was uploaded
        if 'voter_card' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded"
            }), 400

        file = request.files['voter_card']

        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No selected file"
            }), 400

        # Read image bytes
        image_bytes = file.read()
        mime_type = f"image/{file.filename.split('.')[-1]}"

        # Extract details using Gemini API
        extracted_data = get_text_from_image(image_bytes, mime_type)

        if "candidates" in extracted_data and extracted_data["candidates"]:
            extracted_text = extracted_data["candidates"][0]["content"]["parts"][0]["text"]

            # Parse the extracted text into a dictionary
            for line in extracted_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().replace('"', '')
                    value = value.strip().replace('"', '').replace(',', '')
                    details[key] = value

            return jsonify({
                "success": True,
                "message": "Voter card uploaded successfully",
                "details" : details
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Could not extract details from the voter card"
            }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/enter-voter-id', methods=['POST'])
def enter_voter_id():
    try:
        data = request.get_json()
        vid = data.get('voter_id', '').strip()
        voter_details = data.get('voter_details', {})

        # Verify details
        for voter in VOTER_DATA:
            if (voter['vid'] == vid and
                    voter['name'].lower() == details["Name"].strip().lower() and
                    voter['dob'] == details["Date Of Birth"].strip()):
                return jsonify({
                    "message": "Voter ID verified successfully",
                    "verified": True,
                    "details" : details
                }), 200

        return jsonify({
            "message": "Voter ID verification failed",
            "verified": False
        }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/upload-fingerprint', methods=['POST'])
def upload_fingerprint():
    try:
        # Extract Voter ID from form data
        voter_id = request.form.get("voter_id")
        if not voter_id:
            return jsonify({"error": "Voter ID is missing"}), 400

        # Extract fingerprint image from form data
        file = request.files.get('fingerprint')
        if not file:
            return jsonify({"error": "No fingerprint image uploaded"}), 400

        # Convert image file to NumPy array
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        sample = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        # Perform fingerprint matching
        is_match, match_score = match_fingerprint(sample, voter_id)

        return jsonify({
            "message": "Fingerprint verified successfully" if is_match else "Fingerprint verification failed",
            "verified": is_match,
            "match_score": match_score if is_match else None
        }), 200 if is_match else 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
