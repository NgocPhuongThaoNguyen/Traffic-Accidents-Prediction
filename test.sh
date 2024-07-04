host=$1
year=$2
month=$3

curl -s -X POST http://${host}:5000/predict -H "Content-Type: application/json" -d "{\"year\": ${year}, \"month\": ${month}}"
