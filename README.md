# RehabApp

uvicorn main:app --reload  
npm start

docker-compose up --build

docker-compose down
docker-compose --build --no-cache
docker-compose up

pip freeze > requirements.txt

cd rehab-app-backend 
cd rehab-app-frontend

pip list --format=freeze --not-required > requirements.txt

