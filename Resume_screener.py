import streamlit as st
import joblib
import os
import re
import pandas as pd
from textblob import TextBlob
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced text extraction utilities
class TextExtractor:
    """Enhanced text extraction with better error handling"""
    
    @staticmethod
    def extract_text_from_file(uploaded_file) -> Tuple[str, str]:
        """
        Extract text from uploaded file with detailed error reporting
        Returns: (extracted_text, error_message)
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                # Method 1: Try pdfplumber first (more reliable)
                try:
                    import pdfplumber
                    import io
                    
                    # Reset file pointer and read as bytes
                    uploaded_file.seek(0)
                    pdf_bytes = uploaded_file.read()
                    
                    text = ""
                    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    if text.strip():
                        return text, ""
                        
                except ImportError:
                    pass  # Fall back to textract
                except Exception as e:
                    logger.warning(f"pdfplumber failed: {str(e)}, trying textract...")
                
                # Method 2: Fall back to textract with temporary file
                try:
                    import textract
                    
                    # Create temporary file
                    uploaded_file.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        text = textract.process(tmp_file_path).decode('utf-8')
                        return text, ""
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        
                except ImportError:
                    return "", "PDF processing libraries not available. Please install pdfplumber or textract."
                except Exception as e:
                    return "", f"Error processing PDF: {str(e)}"
                    
            elif file_extension == 'docx':
                try:
                    import docx2txt
                    
                    # Create temporary file for docx processing
                    uploaded_file.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        text = docx2txt.process(tmp_file_path)
                        return text, ""
                    finally:
                        os.unlink(tmp_file_path)
                        
                except ImportError:
                    # Alternative method using python-docx
                    try:
                        from docx import Document
                        import io
                        
                        uploaded_file.seek(0)
                        doc = Document(io.BytesIO(uploaded_file.read()))
                        text = ""
                        for paragraph in doc.paragraphs:
                            text += paragraph.text + "\n"
                        return text, ""
                    except ImportError:
                        return "", "DOCX processing library not available. Please install docx2txt or python-docx."
                except Exception as e:
                    return "", f"Error processing DOCX: {str(e)}"
                    
            elif file_extension == 'txt':
                try:
                    uploaded_file.seek(0)
                    text = uploaded_file.read().decode('utf-8')
                    return text, ""
                except UnicodeDecodeError:
                    try:
                        uploaded_file.seek(0)
                        text = uploaded_file.read().decode('latin-1')
                        return text, ""
                    except Exception as e:
                        return "", f"Error reading text file: {str(e)}"
                        
            else:
                return "", f"Unsupported file format: {file_extension}"
                
        except Exception as e:
            logger.error(f"Unexpected error in text extraction: {str(e)}")
            return "", f"Unexpected error: {str(e)}"
        
        # Final validation
        if not text or len(text.strip()) < 50:
            return "", "File appears to be empty or too short to process."
            
        return text, ""

# Enhanced skill extraction
class SkillExtractor:
    """Enhanced skill extraction with comprehensive skill database"""
    
    def __init__(self):
        self.skill_categories = {
            "Programming Languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", "php", 
                "ruby", "go", "rust", "kotlin", "swift", "scala", "r", "matlab",
                "perl", "shell", "bash", "powershell"
            ],
            "Web Technologies": [
                "html", "css", "react", "angular", "vue", "node.js", "express",
                "django", "flask", "fastapi", "spring", "laravel", "rails",
                "bootstrap", "tailwind", "sass", "less", "webpack", "jquery"
            ],
            "Databases": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                "sqlite", "oracle", "sql server", "cassandra", "dynamodb",
                "neo4j", "firebase"
            ],
            "Cloud & DevOps": [
                "aws", "azure", "gcp", "docker", "kubernetes", "jenkins",
                "gitlab", "github", "terraform", "ansible", "chef", "puppet",
                "vagrant", "ci/cd", "devops"
            ],
            "Data Science & ML": [
                "machine learning", "deep learning", "tensorflow", "pytorch",
                "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
                "jupyter", "tableau", "power bi", "spark", "hadoop", "kafka"
            ],
            "Mobile Development": [
                "android", "ios", "react native", "flutter", "xamarin", "ionic",
                "swift", "objective-c", "kotlin"
            ],
            "Other Technologies": [
                "git", "rest api", "graphql", "microservices", "agile", "scrum",
                "jira", "confluence", "slack", "linux", "unix", "windows"
            ]
        }
        
        # Flatten all skills for quick searching
        self.all_skills = []
        for category, skills in self.skill_categories.items():
            self.all_skills.extend(skills)
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills by category from text"""
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skill_categories.items():
            category_skills = []
            for skill in skills:
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    category_skills.append(skill.title())
            
            if category_skills:
                found_skills[category] = category_skills
                
        return found_skills

# Enhanced resume analyzer
class ResumeAnalyzer:
    """Enhanced resume analysis with multiple features"""
    
    def __init__(self, model_path: str = "resume_classifier_model.sav"):
        self.model_path = model_path
        self.model = None
        self.skill_extractor = SkillExtractor()
        self.load_model()
    
    def load_model(self):
        """Load the ML model with error handling"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze text sentiment with detailed scores"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                label = "Positive"
            elif polarity < -0.1:
                label = "Negative"
            else:
                label = "Neutral"
                
            return {
                "label": label,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "confidence": abs(polarity)
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"label": "Unknown", "polarity": 0, "subjectivity": 0, "confidence": 0}
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume"""
        contact_info = {}
        
        # Email extraction
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone extraction
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone'] = ''.join(phones[0]) if isinstance(phones[0], tuple) else phones[0]
        
        # LinkedIn extraction
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text, re.IGNORECASE)
        if linkedin:
            contact_info['linkedin'] = linkedin[0]
            
        return contact_info
    
    def calculate_resume_score(self, text: str, skills_found: Dict) -> int:
        """Calculate an overall resume score based on various factors"""
        score = 0
        
        # Length score (optimal range: 400-1500 words)
        word_count = len(text.split())
        if 400 <= word_count <= 1500:
            score += 20
        elif 200 <= word_count < 400 or 1500 < word_count <= 2000:
            score += 10
        
        # Skills score
        total_skills = sum(len(skills) for skills in skills_found.values())
        if total_skills >= 10:
            score += 30
        elif total_skills >= 5:
            score += 20
        elif total_skills > 0:
            score += 10
        
        # Contact info score
        contact_info = self.extract_contact_info(text)
        score += len(contact_info) * 10
        
        # Experience keywords score
        experience_keywords = ["experience", "worked", "developed", "managed", "led", "created", "implemented"]
        exp_count = sum(1 for keyword in experience_keywords if keyword.lower() in text.lower())
        score += min(exp_count * 5, 20)
        
        # Education keywords score
        education_keywords = ["degree", "university", "college", "bachelor", "master", "phd", "graduate"]
        edu_count = sum(1 for keyword in education_keywords if keyword.lower() in text.lower())
        score += min(edu_count * 3, 15)
        
        return min(score, 100)  # Cap at 100
    
    def analyze_resume(self, text: str) -> Dict:
        """Comprehensive resume analysis"""
        analysis = {}
        
        # Basic info
        analysis['word_count'] = len(text.split())
        analysis['char_count'] = len(text)
        
        # Model prediction
        if self.model:
            try:
                prediction = self.model.predict([text])[0]
                # Try to get prediction probabilities if available
                try:
                    probabilities = self.model.predict_proba([text])[0]
                    max_prob = max(probabilities)
                    analysis['prediction'] = {
                        'role': prediction,
                        'confidence': max_prob
                    }
                except:
                    analysis['prediction'] = {
                        'role': prediction,
                        'confidence': 0.5  # Default confidence
                    }
            except Exception as e:
                logger.error(f"Error in model prediction: {str(e)}")
                analysis['prediction'] = {'role': 'Unknown', 'confidence': 0}
        else:
            analysis['prediction'] = {'role': 'Model not available', 'confidence': 0}
        
        # Skills analysis
        analysis['skills'] = self.skill_extractor.extract_skills(text)
        
        # Sentiment analysis
        analysis['sentiment'] = self.analyze_sentiment(text)
        
        # Contact information
        analysis['contact_info'] = self.extract_contact_info(text)
        
        # Overall score
        analysis['score'] = self.calculate_resume_score(text, analysis['skills'])
        
        return analysis

# Streamlit App
def main():
    st.set_page_config(
        page_title="Smart Resume Screening Tool",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .skill-tag {
        display: inline-block;
        background-color: #e1f5fe;
        color: #01579b;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📄 Smart Resume Screening Tool</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Analysis Options")
        show_detailed_analysis = st.checkbox("Show Detailed Analysis", value=True)
        show_skills_breakdown = st.checkbox("Show Skills Breakdown", value=True)
        show_contact_info = st.checkbox("Show Contact Information", value=True)
        
        st.header("📈 Statistics")
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        if st.session_state.analysis_history:
            avg_score = sum(a['score'] for a in st.session_state.analysis_history) / len(st.session_state.analysis_history)
            st.metric("Average Resume Score", f"{avg_score:.1f}/100")
            st.metric("Total Resumes Analyzed", len(st.session_state.analysis_history))
    
    # Main content
    st.markdown("### Upload Resume for Analysis")
    st.markdown("Supported formats: PDF, DOCX, TXT (Max size: 10MB)")
    
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=["pdf", "docx", "txt"],
        help="Upload a resume file for comprehensive analysis"
    )
    
    if uploaded_file:
        # File validation
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
            st.error("File size too large. Please upload a file smaller than 10MB.")
            return
        
        # Extract text
        with st.spinner("Extracting text from file..."):
            text, error_msg = TextExtractor.extract_text_from_file(uploaded_file)
        
        if error_msg:
            st.error(f"Error processing file: {error_msg}")
            return
        
        if not text:
            st.error("No text could be extracted from the file.")
            return
        
        # Initialize analyzer
        analyzer = ResumeAnalyzer()
        
        # Perform analysis
        with st.spinner("Analyzing resume..."):
            analysis = analyzer.analyze_resume(text)
        
        # Store in history
        analysis['filename'] = uploaded_file.name
        analysis['timestamp'] = datetime.now()
        st.session_state.analysis_history.append(analysis)
        
        # Display results
        st.success("✅ Resume analysis completed!")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Score",
                f"{analysis['score']}/100",
                delta=None,
                help="Overall resume quality score based on multiple factors"
            )
        
        with col2:
            role = analysis['prediction']['role']
            confidence = analysis['prediction']['confidence']
            st.metric(
                "Predicted Role",
                role,
                delta=f"{confidence:.1%} confidence" if confidence > 0 else None
            )
        
        with col3:
            sentiment_info = analysis['sentiment']
            st.metric(
                "Sentiment",
                sentiment_info['label'],
                delta=f"{sentiment_info['polarity']:.2f}" if sentiment_info['polarity'] != 0 else None
            )
        
        with col4:
            total_skills = sum(len(skills) for skills in analysis['skills'].values())
            st.metric("Skills Found", total_skills)
        
        # Detailed analysis sections
        if show_detailed_analysis:
            st.markdown("---")
            st.markdown("### 📋 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Document Statistics:**")
                st.write(f"• Word Count: {analysis['word_count']}")
                st.write(f"• Character Count: {analysis['char_count']}")
                st.write(f"• Estimated Reading Time: {analysis['word_count'] // 200 + 1} minutes")
            
            with col2:
                st.markdown("**Sentiment Analysis:**")
                sentiment = analysis['sentiment']
                st.write(f"• Overall Tone: {sentiment['label']}")
                st.write(f"• Polarity Score: {sentiment['polarity']:.3f}")
                st.write(f"• Subjectivity: {sentiment['subjectivity']:.3f}")
        
        # Skills breakdown
        if show_skills_breakdown and analysis['skills']:
            st.markdown("---")
            st.markdown("### 🛠️ Skills Analysis")
            
            for category, skills in analysis['skills'].items():
                st.markdown(f"**{category}:**")
                skills_html = " ".join([f'<span class="skill-tag">{skill}</span>' for skill in skills])
                st.markdown(skills_html, unsafe_allow_html=True)
                st.markdown("")
        elif show_skills_breakdown:
            st.markdown("---")
            st.warning("No matching skills found in the resume. Consider adding more technical skills.")
        
        # Contact information
        if show_contact_info and analysis['contact_info']:
            st.markdown("---")
            st.markdown("### 📞 Contact Information")
            
            contact = analysis['contact_info']
            for key, value in contact.items():
                st.write(f"• {key.title()}: {value}")
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        recommendations = []
        
        if analysis['score'] < 50:
            recommendations.append("Consider adding more relevant skills and experience details")
        
        if analysis['word_count'] < 300:
            recommendations.append("Resume appears too short. Add more details about your experience")
        elif analysis['word_count'] > 1500:
            recommendations.append("Resume might be too long. Consider condensing to 1-2 pages")
        
        if not analysis['contact_info'].get('email'):
            recommendations.append("Make sure to include a professional email address")
        
        if len(analysis['skills']) < 3:
            recommendations.append("Include more technical skills relevant to your field")
        
        if analysis['sentiment']['polarity'] < 0:
            recommendations.append("Consider using more positive language to describe your achievements")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("Great resume! No major improvements needed.")
        
        # Download analysis report
        st.markdown("---")
        if st.button("📥 Generate Analysis Report"):
            report = generate_analysis_report(analysis, uploaded_file.name)
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"resume_analysis_{uploaded_file.name.split('.')[0]}.txt",
                mime="text/plain"
            )

def generate_analysis_report(analysis: Dict, filename: str) -> str:
    """Generate a text report of the analysis"""
    report = f"""
RESUME ANALYSIS REPORT
=====================
File: {filename}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL SCORE: {analysis['score']}/100

PREDICTED JOB ROLE:
Role: {analysis['prediction']['role']}
Confidence: {analysis['prediction']['confidence']:.1%}

DOCUMENT STATISTICS:
Word Count: {analysis['word_count']}
Character Count: {analysis['char_count']}

SENTIMENT ANALYSIS:
Overall Tone: {analysis['sentiment']['label']}
Polarity Score: {analysis['sentiment']['polarity']:.3f}
Subjectivity: {analysis['sentiment']['subjectivity']:.3f}

SKILLS FOUND:
"""
    
    for category, skills in analysis['skills'].items():
        report += f"\n{category}:\n"
        for skill in skills:
            report += f"  - {skill}\n"
    
    if analysis['contact_info']:
        report += "\nCONTACT INFORMATION:\n"
        for key, value in analysis['contact_info'].items():
            report += f"{key.title()}: {value}\n"
    
    return report

if __name__ == "__main__":
    main()
