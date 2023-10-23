# End to end machine learning: MLOps Recommendation system

## Summary:

This repository demonstrates how to deploy an end-to-end ML application using **CI/CD pipelines** and GitHub Actions, in combination with a container registry and Azure Web App. And provides a hands-on approach to deploying ML models, making it easier for both beginners developers to embrace this technology. By using CI/CD pipelines, GitHub Actions, a container registry, and Azure Web App, you can streamline the deployment process, ensuring that your machine learning models are always up to date and readily accessible.


## Getting Started 💡
The data I use is [Amazon Sales Datasets](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset). I only use a few basic properties of the dataset, you can explore it further yourself
1. Data Ingestion :
  - In the Data Ingestion phase, the initial step involves reading the data from a CSV file.
  - Subsequently, the data is partitioned into training and testing sets, which are then saved as CSV files.
2. Data Transformation :
 - Preprocess data with scaling and encoding, saving as a PKL file.
3. Model Training :
 - Train, evaluate, and chosee the best model.
4. Prediction Pipeline :
 - Utilize pickle files for predictions in a Python environment.
5. Flask App creation :
 - Create web app
     
### Run in localhost:

```zsh
git clone https://github.com/Harly-1506/MLOps-Recommendation.git
python -m venv venv
source venv/bin/activate
#test training
python setup.py install
python src/components/data_ingestion.py 
```

Then you have to create a Docker image, Container Registry and Azure Web App and run:
```zsh
docker build -t <registry>.azurecr.io/<name>:latest .

docker login <registry>.azurecr.io

docker push <registry>.azurecr.io/<name>:latest

```
## Build your own project

In this project, I've only established the fundamental components. You can explore additional enhancements, such as Optimizing Model Parameters, Advanced Data Processing, Implement Feature Engineering, Continuous Integration. By incorporating these ideas and utilizing DVC for version control and data management, you can take your project to the next level, making it more robust, adaptable, and efficient.

---
*Author: Harly*
