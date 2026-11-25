
from flask import Flask, request, jsonify
from convert_handler import ConvertHandler

app = Flask(__name__)
handler = ConvertHandler()


@app.route("/")
def home():
    return "Metric-Imperial Converter API is running!"


@app.route("/api/convert", methods=["GET"])
def convert_api():

    input_string = request.args.get("input", "")

    number, unit = handler.parse_input(input_string)

    result = handler.convert(number, unit)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=3000)
