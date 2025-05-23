import os
import re
import logging
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import joblib
from datetime import datetime

# Text extraction imports with fallback handling
try:
    import docx2txt
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("docx2txt not available. DOCX processing will be disabled.")

try:
    import textract
    PDF_TEXTRACT_AVAILABLE = True
except ImportError:
    PDF_TEXTRACT_AVAILABLE = False
    logging.warning("textract not available. PDF processing via textract will be disabled.")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available. Alternative PDF processing will be disabled.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logging.warning("pdfplumber not available. Alternative PDF processing will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    """Enhanced text extraction with multiple fallback methods"""
    
    @staticmethod
    def extract_text_from_pdf_textract(file_path: str) -> Tuple[str, str]:
        """Extract text using textract library"""
        try:
            if not PDF_TEXTRACT_AVAILABLE:
                return "", "textract library not available"
            
            text = textract.process(file_path, extension='pdf').decode('utf-8')
            return text, ""
        except Exception as e:
            return "", f"textract error: {str(e)}"
    
    @staticmethod
    def extract_text_from_pdf_pypdf2(file_path: str) -> Tuple[str, str]:
        """Extract text using PyPDF2 library"""
        try:
            if not PYPDF2_AVAILABLE:
                return "", "PyPDF2 library not available"
            
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            return text, ""
        except Exception as e:
            return "", f"PyPDF2 error: {str(e)}"
    
    @staticmethod
    def extract_text_from_pdf_pdfplumber(file_path: str) -> Tuple[str, str]:
        """Extract text using pdfplumber library"""
        try:
            if not PDFPLUMBER_AVAILABLE:
                return "", "pdfplumber library not available"
            
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text, ""
        except Exception as e:
            return "", f"pdfplumber error: {str(e)}"
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Tuple[str, str]:
        """Extract text from PDF using multiple fallback methods"""
        methods = [
            TextExtractor.extract_text_from_pdf_textract,
            TextExtractor.extract_text_from_pdf_pdfplumber,
            TextExtractor.extract_text_from_pdf_pypdf2,
        ]
        
        errors = []
        for method in methods:
            text, error = method(file_path)
            if text and len(text.strip()) > 20:  # Minimum viable text length
                return text, ""
            if error:
                errors.append(error)
        
        return "", f"All PDF extraction methods failed: {'; '.join(errors)}"
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> Tuple[str, str]:
        """Extract text from DOCX file"""
        try:
            if not DOCX_AVAILABLE:
                return "", "docx2txt library not available"
            
            text = docx2txt.process(file_path)
            if not text or len(text.strip()) < 10:
                return "", "No meaningful text extracted from DOCX"
            
            return text, ""
        except Exception as e:
            return "", f"DOCX extraction error: {str(e)}"
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> Tuple[str, str]:
        """Extract text from TXT file with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                    if text and len(text.strip()) > 10:
                        return text, ""
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                return "", f"TXT extraction error: {str(e)}"
        
        return "", "Could not decode text file with any supported encoding"
    
    @staticmethod
    def extract_text(file_path: str) -> Tuple[str, str]:
        """
        Main text extraction method with comprehensive error handling
        Returns: (extracted_text, error_message)
        """
        if not os.path.exists(file_path):
            return "", "File does not exist"
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return "", "File is empty"
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return "", "File too large (>50MB)"
        
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return TextExtractor.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                return TextExtractor.extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                return TextExtractor.extract_text_from_txt(file_path)
            else:
                return "", f"Unsupported file format: {file_extension}"
                
        except Exception as e:
            logger.error(f"Unexpected error in text extraction: {str(e)}")
            return "", f"Unexpected error: {str(e)}"

class AdvancedSkillExtractor:
    """Advanced skill extraction with comprehensive skill database and NLP techniques"""
    
    def __init__(self):
        self.skill_database = self._load_skill_database()
        self.skill_patterns = self._compile_skill_patterns()
    
    def _load_skill_database(self) -> Dict[str, List[str]]:
        """Load comprehensive skill database"""
        return {
            "Programming Languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", "c",
                "php", "ruby", "go", "rust", "kotlin", "swift", "scala", "r",
                "matlab", "perl", "shell", "bash", "powershell", "vb.net",
                "assembly", "cobol", "fortran", "haskell", "lua", "dart",
                "objective-c", "pascal", "prolog", "scheme", "smalltalk"
            ],
            "Web Technologies": [
                "html", "css", "html5", "css3", "sass", "scss", "less", "stylus",
                "react", "angular", "vue", "vue.js", "svelte", "ember", "backbone",
                "node.js", "express", "express.js", "koa", "nest.js", "next.js",
                "nuxt.js", "gatsby", "django", "flask", "fastapi", "tornado",
                "spring", "spring boot", "laravel", "symfony", "codeigniter",
                "rails", "ruby on rails", "asp.net", "blazor", "struts",
                "bootstrap", "bulma", "foundation", "tailwind", "semantic ui",
                "material-ui", "ant design", "chakra ui", "jquery", "lodash",
                "webpack", "parcel", "rollup", "vite", "gulp", "grunt",
                "babel", "typescript", "graphql", "rest api", "soap", "grpc"
            ],
            "Mobile Development": [
                "android", "ios", "react native", "flutter", "xamarin", "ionic",
                "cordova", "phonegap", "kotlin", "swift", "objective-c",
                "java android", "android studio", "xcode", "unity 3d"
            ],
            "Databases": [
                "mysql", "postgresql", "sqlite", "mongodb", "redis", "cassandra",
                "elasticsearch", "solr", "neo4j", "dynamodb", "couchdb",
                "oracle", "sql server", "mariadb", "firestore", "firebase",
                "influxdb", "cockroachdb", "etcd", "memcached", "hbase",
                "clickhouse", "snowflake", "bigquery", "redshift"
            ],
            "Cloud Platforms & Services": [
                "aws", "amazon web services", "azure", "microsoft azure",
                "gcp", "google cloud", "google cloud platform", "digitalocean",
                "heroku", "netlify", "vercel", "firebase", "supabase",
                "cloudflare", "linode", "vultr", "rackspace", "alibaba cloud",
                "oracle cloud", "ibm cloud", "salesforce", "vmware",
                "openstack", "kubernetes", "docker", "docker compose"
            ],
            "DevOps & Infrastructure": [
                "devops", "ci/cd", "jenkins", "gitlab ci", "github actions",
                "azure devops", "circleci", "travis ci", "bamboo", "teamcity",
                "docker", "kubernetes", "k8s", "helm", "istio", "terraform",
                "ansible", "chef", "puppet", "vagrant", "packer", "consul",
                "vault", "nomad", "nginx", "apache", "haproxy", "envoy",
                "prometheus", "grafana", "elk", "elasticsearch", "logstash",
                "kibana", "splunk", "datadog", "new relic", "nagios", "zabbix"
            ],
            "Data Science & Analytics": [
                "data science", "machine learning", "deep learning", "ai",
                "artificial intelligence", "neural networks", "tensorflow",
                "pytorch", "keras", "scikit-learn", "pandas", "numpy",
                "matplotlib", "seaborn", "plotly", "bokeh", "jupyter",
                "ipython", "anaconda", "r", "rstudio", "tableau", "power bi",
                "qlik", "looker", "d3.js", "apache spark", "hadoop", "hive",
                "pig", "kafka", "storm", "flink", "airflow", "luigi",
                "mlflow", "kubeflow", "dvc", "weights & biases", "tensorboard"
            ],
            "Testing & Quality Assurance": [
                "testing", "unit testing", "integration testing", "selenium",
                "cypress", "jest", "mocha", "jasmine", "karma", "protractor",
                "testng", "junit", "pytest", "unittest", "rspec", "cucumber",
                "postman", "insomnia", "soapui", "jmeter", "loadrunner",
                "k6", "artillery", "locust", "gatling", "appium", "detox"
            ],
            "Version Control & Collaboration": [
                "git", "github", "gitlab", "bitbucket", "mercurial", "svn",
                "perforce", "tfs", "azure repos", "codecommit", "sourcetree",
                "gitkraken", "fork", "tower", "smartgit", "tortoisegit"
            ],
            "Project Management & Methodologies": [
                "agile", "scrum", "kanban", "lean", "waterfall", "devops",
                "jira", "confluence", "trello", "asana", "monday.com",
                "notion", "slack", "microsoft teams", "discord", "zoom",
                "azure boards", "github projects", "linear", "clickup"
            ],
            "Operating Systems & Tools": [
                "linux", "unix", "windows", "macos", "ubuntu", "centos",
                "debian", "redhat", "fedora", "suse", "arch", "freebsd",
                "vim", "emacs", "vscode", "intellij", "eclipse", "netbeans",
                "sublime text", "atom", "brackets", "notepad++", "pycharm",
                "webstorm", "android studio", "xcode", "visual studio"
            ],
            "Security & Networking": [
                "cybersecurity", "information security", "penetration testing",
                "ethical hacking", "vulnerability assessment", "owasp",
                "ssl", "tls", "https", "oauth", "jwt", "saml", "ldap",
                "active directory", "kerberos", "ipsec", "vpn", "firewall",
                "ids", "ips", "siem", "wireshark", "nmap", "metasploit",
                "burp suite", "nessus", "qualys", "rapid7"
            ],
            "Design & UI/UX": [
                "ui", "ux", "user interface", "user experience", "design",
                "graphic design", "web design", "figma", "sketch", "adobe xd",
                "invision", "zeplin", "marvel", "principle", "framer",
                "photoshop", "illustrator", "after effects", "premiere pro",
                "canva", "gimp", "inkscape", "blender", "3d modeling"
            ],
            "Business & Analytics": [
                "business analysis", "product management", "data analysis",
                "excel", "google sheets", "vba", "pivot tables", "vlookup",
                "sql", "business intelligence", "etl", "data warehousing",
                "crm", "erp", "salesforce", "hubspot", "marketo", "mailchimp",
                "google analytics", "adobe analytics", "mixpanel", "amplitude"
            ]
        }
    
    def _compile_skill_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for better skill matching"""
        patterns = {}
        
        for category, skills in self.skill_database.items():
            category_patterns = []
            for skill in skills:
                # Create word boundary pattern for exact matches
                escaped_skill = re.escape(skill.lower())
                # Handle special cases like C++ and C#
                if skill.lower() in ['c++', 'c#']:
                    pattern = r'\b' + escaped_skill + r'\b'
                else:
                    pattern = r'\b' + escaped_skill + r'\b'
                category_patterns.append(pattern)
            
            # Combine all patterns for this category
            combined_pattern = '|'.join(category_patterns)
            patterns[category] = re.compile(combined_pattern, re.IGNORECASE)
        
        return patterns
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills using compiled regex patterns"""
        text_lower = text.lower()
        found_skills = {}
        
        for category, pattern in self.skill_patterns.items():
            matches = pattern.findall(text_lower)
            if matches:
                # Remove duplicates and format properly
                unique_matches = list(set(matches))
                formatted_skills = []
                
                for match in unique_matches:
                    # Find the original skill name from database
                    for skill in self.skill_database[category]:
                        if skill.lower() == match.lower():
                            formatted_skills.append(skill)
                            break
                
                if formatted_skills:
                    found_skills[category] = sorted(formatted_skills)
        
        return found_skills
    
    def get_skill_recommendations(self, current_skills: Dict[str, List[str]], target_role: str = None) -> List[str]:
        """Suggest additional skills based on current skills and target role"""
        recommendations = []
        
        # Role-based recommendations
        role_skills = {
            "data scientist": ["python", "r", "tensorflow", "pandas", "jupyter", "sql"],
            "web developer": ["javascript", "react", "node.js", "html", "css", "git"],
            "devops engineer": ["docker", "kubernetes", "aws", "jenkins", "terraform", "linux"],
            "mobile developer": ["react native", "flutter", "kotlin", "swift", "android", "ios"],
            "backend developer": ["python", "java", "sql", "redis", "docker", "api"],
            "frontend developer": ["javascript", "react", "vue", "html", "css", "webpack"]
        }
        
        if target_role and target_role.lower() in role_skills:
            current_skill_list = [skill.lower() for skills in current_skills.values() for skill in skills]
            for recommended_skill in role_skills[target_role.lower()]:
                if recommended_skill not in current_skill_list:
                    recommendations.append(recommended_skill.title())
        
        return recommendations[:5]  # Limit to top 5 recommendations

class ModelTrainer:
    """Enhanced model training with better data handling and validation"""
    
    def __init__(self, model_save_path: str = "resume_classifier_model.sav"):
        self.model_save_path = model_save_path
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.training_history = []
    
    def prepare_training_data(self, data_path: str) -> Tuple[List[str], List[str]]:
        """Load and prepare training data from various formats"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
                df = pd.read_excel(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            # Assume columns are 'text' and 'label' or similar
            text_columns = ['text', 'resume_text', 'content', 'description']
            label_columns = ['label', 'category', 'job_role', 'position']
            
            text_col = None
            label_col = None
            
            for col in df.columns:
                if col.lower() in text_columns:
                    text_col = col
                if col.lower() in label_columns:
                    label_col = col
            
            if not text_col or not label_col:
                raise ValueError("Could not identify text and label columns in the dataset")
            
            # Clean the data
            df = df.dropna(subset=[text_col, label_col])
            df[text_col] = df[text_col].astype(str)
            df[label_col] = df[label_col].astype(str)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=[text_col])
            
            return df[text_col].tolist(), df[label_col].tolist()
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def train_model(self, texts: List[str], labels: List[str], model_type: str = "random_forest"):
        """Train the classification model with cross-validation"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Prepare the data
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            X = self.vectorizer.fit_transform(texts)
            
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(labels)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Choose model
            models = {
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "svm": SVC(kernel='rbf', probability=True, random_state=42),
                "naive_bayes": MultinomialNB(),
                "logistic_regression": LogisticRegression(random_state=42, max_iter=1000)
            }
            
            if model_type not in models:
                model_type = "random_forest"
            
            self.model = models[model_type]
            
            # Train the model
            logger.info(f"Training {model_type} model...")
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            
            # Predictions for detailed evaluation
            y_pred = self.model.predict(X_test)
            
            # Store training history
            training_info = {
                "timestamp": datetime.now(),
                "model_type": model_type,
                "train_score": train_score,
                "test_score": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "num_samples": len(texts),
                "num_classes": len(np.unique(labels)),
                "classification_report": classification_report(y_test, y_pred, 
                                                             target_names=self.label_encoder.classes_),
                "feature_names": self.vectorizer.get_feature_names_out()[:100].tolist()
            }
            
            self.training_history.append(training_info)
            
            logger.info(f"Model trained successfully!")
            logger.info(f"Training accuracy: {train_score:.4f}")
            logger.info(f"Test accuracy: {test_score:.4f}")
            logger.info(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return training_info
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self):
        """Save the trained model and associated components"""
        try:
            if self.model is None:
                raise ValueError("No model to save. Train a model first.")
            
            model_data = {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label_encoder': self.label_encoder,
                'training_history': self.training_history,
                'save_timestamp': datetime.now()
            }
            
            joblib.dump(model_data, self.model_save_path)
            logger.info(f"Model saved to {self.model_save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self):
        """Load a previously trained model"""
        try:
            if not os.path.exists(self.model_save_path):
                raise FileNotFoundError(f"Model file not found: {self.model_save_path}")
            
            model_data = joblib.load(self.model_save_path)
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.label_encoder = model_data['label_encoder']
            self.training_history = model_data.get('training_history', [])
            
            logger.info(f"Model loaded from {self.model_save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make predictions with confidence scores"""
        try:
            if self.model is None or self.vectorizer is None:
                raise ValueError("Model not loaded. Load or train a model first.")
            
            X = self.vectorizer.transform(texts)
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                predicted_label = self.label_encoder.inverse_transform([pred])[0]
                confidence = max(probs)
                
                # Get top 3 predictions
                top_indices = np.argsort(probs)[-3:][::-1]
                top_predictions = []
                
                for idx in top_indices:
                    label = self.label_encoder.inverse_transform([idx])[0]
                    prob = probs[idx]
                    top_predictions.append({"label": label, "probability": prob})
                
                results.append({
                    "text_index": i,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "top_predictions": top_predictions
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

def train_and_save_model(data_path: str = None, model_type: str = "random_forest"):
    """Convenience function to train and save a model"""
    try:
        if not data_path:
            logger.error("No training data path provided")
            return False
        
        trainer = ModelTrainer()
        texts, labels = trainer.prepare_training_data(data_path)
        
        logger.info(f"Loaded {len(texts)} training samples with {len(set(labels))} unique labels")
        
        training_info = trainer.train_model(texts, labels, model_type)
        trainer.save_model()
        
        return training_info
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        return False

def extract_text(uploaded_file) -> str:
    """Legacy function for backward compatibility"""
    if hasattr(uploaded_file, 'name'):
        # Streamlit uploaded file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            text, error = TextExtractor.extract_text(tmp_file_path)
            return text if not error else ""
        finally:
            os.unlink(tmp_file_path)
    else:
        # File path
        text, error = TextExtractor.extract_text(uploaded_file)
        return text if not error else ""

def extract_skills(text: str) -> List[str]:
    """Legacy function for backward compatibility"""
    extractor = AdvancedSkillExtractor()
    skills_dict = extractor.extract_skills(text)
    
    # Flatten all skills into a single list
    all_skills = []
    for category_skills in skills_dict.values():
        all_skills.extend([skill.lower() for skill in category_skills])
    
    return all_skills

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else "random_forest"
        
        print(f"Training model with data from: {data_path}")
        result = train_and_save_model(data_path, model_type)
        
        if result:
            print("Model training completed successfully!")
            print(f"Model type: {result['model_type']}")
            print(f"Test accuracy: {result['test_score']:.4f}")
            print(f"Cross-validation score: {result['cv_mean']:.4f}")
        else:
            print("Model training failed!")
    else:
        print("Usage: python models_utils.py <data_path> [model_type]")
        print("Example: python models_utils.py training_data.csv random_forest")