from wende import create_app
from wende.config import DEBUG

wende = create_app()

if __name__ == '__main__':
    wende.run(host='127.0.0.1', port=9191, debug=DEBUG)
