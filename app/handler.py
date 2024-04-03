from flask import Flask, request, jsonify
from model import predict_image 
import os
import json
import boto3
import logging

s3 = boto3.client('s3')

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['POST'])
def predict():
    app.logger.info("Request Headers:")
    for header, value in request.headers.items():
        app.logger.info(f"{header}: {value}")

    json_data = request.get_json(silent=True)
    if json_data is None or 'Records' not in json_data:
        app.logger.info("No JSON data could be parsed or 'Records' key is missing.")
        return jsonify({'error': 'No valid JSON data found or missing "Records" key.'}), 400
    
    app.logger.info("JSON Data:")
    app.logger.info(json.dumps(json_data, indent=2))
    
    # maybe not needed?
    json_data = json.loads(request.json)
    records = json_data['Records']
    
    for record in records:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        s3.download_file(bucket, key, key)
        app.logger.info(f"Downloaded file: {key}")
        
        prediction_result = predict_image('pc_model.pth', key)
        app.logger.info(prediction_result)
        
        result_string = json.dumps(prediction_result)
        
        result_bucket = os.getenv('RESULT_BUCKET')
        result_key = 'predictions/' + os.path.splitext(key)[0] + '_result.json'
        
        s3.put_object(Body=result_string, Bucket=result_bucket, Key=result_key)
        app.logger.info(f"Uploaded prediction result for {key} to {result_bucket}/{result_key}")
        
        os.remove(key)

    return jsonify({'status': 'ok'}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)