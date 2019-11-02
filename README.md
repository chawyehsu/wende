Wende
---
Wende(问得) Chinese Question Answering, is a small and simple factoid
chinese question answering system written in Python. You ask a chinese
natural language question and Wende will try to give a certain answer
of the question. This is still a work in progress, currently the system
can only judge the type of the question that the user ask, which means,
only the question classification module has been implemented.

## Run within Docker

1. Install and startup [Docker Desktop](https://docs.docker.com/docker-for-windows/).

2. Clone the source.
   ``` sh
   $ git clone https://github.com/h404bi/wende
   $ cd wende/
   ```

3. Download the models and place them into their coresponding directories.  
   - LTP model: http://ltp.ai/  
   - Wende & Word2Vec: https://pan.baidu.com/s/1nv5ubJr (1dbm)  
   
   P.S. you can also train Wende model by yourself, see `wende/classification/model.py` for more information.

4. Then execute `docker-compose` inside the directory, to build Docker image.
   ``` sh
   $ docker-compose up
   ```

5. Waiting for image building, it will take some times, after building success, open your browser and enter `127.0.0.1:9191`, boom!

6. Play with it.

## License
Code: MIT License, Copyright (c) 2016-present, by Chawye Hsu  
Data: [CC BY-NC-SA 4.0], except for the [ltp model](http://ltp.ai/) use its original license.  
This is just the graduation project of my bachelor's degree. (这只是我的学士学位毕业设计项目)


[CC BY-NC-SA 4.0]: https://creativecommons.org/licenses/by-nc-sa/4.0/
