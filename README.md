# AI-Notebooks
All jupyter notebooks from my 3-month AI & Machine learning practicum program at Lateral.

## Assignment Overview

### [A1: Python Fundamentals - To-Do List Manager](A1/Assignment1.ipynb)
Introduction to Python programming fundamentals. Built a command-line to-do list manager implementing basic concepts like lists, functions, loops, and user interaction. Features include adding tasks, viewing tasks, marking tasks complete, and removing tasks with basic error handling.

### [A2: Data Analysis with Pandas - NYC Airbnb Analysis](A2/Assignment2.ipynb)
Explored and analyzed the New York City Airbnb Open Data dataset using Pandas and data visualization libraries. Tasks included handling missing values, extracting insights from the data, analyzing pricing patterns, reviewing trends, and comparing different neighborhoods and room types. Emphasized creative data exploration and drawing meaningful conclusions from real-world data.

### [A3: Data Preprocessing - Titanic Dataset](A3/Assignment3.ipynb)
Comprehensive data preprocessing assignment using the Kaggle Titanic dataset. Covered essential data preparation techniques including:
- Exploratory data analysis with summary statistics and visualizations
- Missing value detection and imputation strategies
- Outlier detection and handling
- Data type identification and transformation
- Comparison of datasets before and after preprocessing

### [A4: Linear Regression - House Price Prediction](A4/Assignment4.ipynb)
Applied linear regression techniques to predict house prices using datasets like Boston Housing or California Housing. Focused on:
- Data exploration and cleaning
- Handling missing/inconsistent values
- Feature-target relationship visualization
- Model building and evaluation
- Performance comparison between different approaches

### [A5: Classification Methods - Heart Disease Prediction](A5/Assignment5.ipynb)
Comprehensive classification challenge using the Heart Disease UCI dataset with 13 clinical features. Applied and compared four different classification methods to predict heart disease status. Tasks included:
- Data exploration and preprocessing
- Missing value handling
- Feature engineering and creation
- Categorical variable encoding
- Train-test splitting
- Model comparison and performance analysis
- Healthcare application interpretation

### [A6: Model Evaluation - Mushroom Classification](A6/Assignment6.ipynb)
Focused on model evaluation techniques using the Mushroom Classification dataset. Practiced:
- Exploratory data analysis (EDA)
- Data preprocessing and feature engineering
- Model training with various algorithms
- Performance evaluation using appropriate metrics
- Cross-validation techniques
- Model comparison and selection

### [A7: Clustering Analysis - Mall Customers](A7/Assignment7.ipynb)
Applied various clustering methods to the Mall Customers dataset. Explored unsupervised learning techniques including:
- Multiple clustering algorithms
- Correlation analysis and interpretation
- Identifying counterintuitive patterns
- Dimensionality reduction for visualization
- Cluster interpretation and business insights
- Bonus: Credit card dataset clustering analysis

### [A8: Neural Networks with PyTorch](A8/Assignment8.ipynb)
Introduction to building and training neural networks using PyTorch. Covered:
- Building feedforward neural networks (MLPs)
- Training neural networks from scratch
- Observing and mitigating underfitting and overfitting
- Comparing performance on regression vs classification tasks
- Understanding training dynamics and optimization

### [A9: Deep Learning - Image Classification Benchmarking](A9/Assignment9.ipynb)
Advanced deep learning assignment focusing on image classification model benchmarking using PyTorch Lightning. Implemented sophisticated training techniques including:
- Custom plant dataset classification
- PyTorch Lightning training framework
- Model checkpointing for saving best models
- Early stopping to prevent overfitting
- Learning rate monitoring and scheduling
- Multi-epoch training with callbacks
- Model testing and evaluation
- Performance comparison across different architectures

### [A10: Transfer Learning - EfficientNet Plant Classification](A10/Assignment10.ipynb)
Applied transfer learning techniques using pre-trained EfficientNet-B0 model for plant species classification. Key components:
- Fine-tuning pre-trained EfficientNet architecture
- Advanced image preprocessing and augmentation
- Model optimization for plant classification
- Saving trained models and preprocessing parameters
- Inference on test images
- Performance evaluation and visualization

### [A11: LLMs, Prompt Engineering & Agents - Plant Classifier Enhancement](A11/Assignment10&11.ipynb)
Enhanced the plant classification system with Large Language Models and intelligent agents. Explored:
- LangChain framework integration
- OpenAI GPT models for plant information generation
- Prompt engineering for structured outputs
- Retrieval Augmented Generation (RAG) with FAISS vector stores
- Web search integration using Tavily
- Agent-based information retrieval
- Combining multiple AI techniques for comprehensive plant care recommendations

## Projects

### [Plant-Classifier: Full-Stack Plant Classification Application](Plant-Classifier/)
A production-ready web application that combines deep learning, RAG, and LLM agents to provide comprehensive plant identification and care recommendations. Features include:

**Backend (FastAPI)**
- EfficientNet-B0 model for plant image classification
- Multiple inference endpoints (basic prediction, RAG-enhanced, LLM-powered)
- Integration with OpenAI GPT for plant care card generation
- RAG system using FAISS vector store for grounded information
- Web search agent using Tavily for real-time plant data
- RESTful API with automatic documentation (Swagger/OpenAPI)
- Health checks and error handling

**Frontend (React)**
- Modern, responsive UI for plant image upload
- Real-time classification results
- Display of detailed plant care recommendations
- Integration with multiple backend endpoints

**Deployment**
- Docker containerization with multi-stage builds
- Docker Compose for orchestration
- Nginx for frontend serving and API proxy
- Environment variable configuration
- Health checks and automatic restarts

**Technologies**: PyTorch, FastAPI, React, LangChain, FAISS, OpenAI API, Tavily API, Docker, Nginx

**How to Run**:
- **Locally**: Use `start_backend.ps1` and `start_frontend.ps1` scripts
- **Docker**: Run `docker-compose up --build -d` from the Plant-Classifier directory
- **Access**: Frontend at http://localhost, Backend API at http://localhost:8080/docs

## Technologies Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning**: scikit-learn
- **Deep Learning**: PyTorch, PyTorch Lightning, EfficientNet
- **LLM & AI Agents**: LangChain, OpenAI API, Tavily API, FAISS
- **Web Development**: FastAPI, React, Nginx
- **Deployment**: Docker, Docker Compose
- **Tools**: Jupyter Notebooks, Kaggle datasets

## Project Structure
Each assignment folder (A1-A11) contains:
- Jupyter notebook with implementation and analysis
- Relevant datasets (CSV files or data directories)
- Supporting files and caches where applicable

The Plant-Classifier project includes:
- Complete full-stack application structure
- Backend API with multiple endpoints
- React frontend application
- Docker configuration files
- Model files and vector stores
- Comprehensive documentation