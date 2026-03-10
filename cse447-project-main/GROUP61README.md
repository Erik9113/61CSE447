# CSE447 GROUP 61 Project

We will train using Wikipedia articles which have millions of articles in different languages.
Wikipedia is free online encyclopedia that has diverse writing styles making it suitable to train
character language models like n-gram.

We will implement a character level language model using n-gram that will predict the next most
likely character from context. Our goal is to implemented in python.

To set up the environment go through the following,
1. Clone repository
2. Start Docker Desktop
3. From project root run:
   docker build -t cse447-project .
   docker run -it -v "$(pwd -W):/job" cse447-project bash
   bash setup.sh
4. Extract text using WikiExtractor
5. Train model (offline)


Train using
```
python src/myprogram.py train --work_dir work
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output pred.txt
python grader/grade.py pred.txt example/answer.txt --verbose
```
