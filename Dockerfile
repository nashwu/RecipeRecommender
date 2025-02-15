FROM python:3.12

WORKDIR /app/backend

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . .

ENV DJANGO_SETTINGS_MODULE=recipe_recommender.settings
ENV PYTHONUNBUFFERED=1

RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "recipe_recommender.wsgi:application"]
