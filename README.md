# RehabApp

vicorn main:app --host 0.0.0.0 --port 8000 --relo
npm start

docker-compose up --build

docker-compose down
docker-compose build --no-cache
docker-compose up

pip freeze > requirements.txt

cd rehab-app-backend 
cd rehab-app-frontend

pip list --format=freeze --not-required > requirements.txt

pyobjc==11.0 delete this package from model training requirements.txt