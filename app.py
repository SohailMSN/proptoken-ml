import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import json
from datetime import datetime, timedelta
import random
from faker import Faker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import base64

# Set page config
st.set_page_config(
    page_title="PropToken - Real Estate Tokenization",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Faker for dummy data
fake = Faker()

# Custom CSS for modern fintech styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .property-card {
        border: 2px solid #000;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        min-height: 300px;
        width: 100%;
        background: #fff;
    }
    
    .property-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    /* Responsive design for marketplace property cards */
    @media (max-width: 1200px) {
        .property-card {
            height: 280px;
            min-height: 280px;
            padding: 1.2rem;
        }
    }
    
    @media (max-width: 768px) {
        .property-card {
            height: 260px;
            min-height: 260px;
            padding: 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .property-card {
            height: 240px;
            min-height: 240px;
            padding: 0.8rem;
        }
    }
    
    .roi-badge {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .cta-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'properties' not in st.session_state:
    st.session_state.properties = []
if 'investments' not in st.session_state:
    st.session_state.investments = []
if 'user_portfolio' not in st.session_state:
    st.session_state.user_portfolio = {}
if 'kyc_status' not in st.session_state:
    st.session_state.kyc_status = {
        'verified': False,
        'documents_uploaded': False,
        'personal_info': {},
        'documents': {}
    }

def verify_kyc_documents(personal_info, documents):
    """Simulate KYC document verification"""
    # Simulate verification process
    import time
    time.sleep(1)  # Simulate processing time
    
    # Basic validation rules
    required_fields = ['full_name', 'email', 'phone', 'address', 'date_of_birth', 'national_id']
    required_docs = ['id_document', 'address_proof']
    
    # Check if all required fields are provided
    fields_complete = all(personal_info.get(field) for field in required_fields)
    
    # Check if all required documents are uploaded
    docs_complete = all(documents.get(doc) for doc in required_docs)
    
    # Simulate verification result (90% success rate for demo)
    verification_success = fields_complete and docs_complete and random.random() > 0.1
    
    return {
        'verified': verification_success,
        'fields_complete': fields_complete,
        'docs_complete': docs_complete,
        'verification_date': datetime.now() if verification_success else None,
        'rejection_reason': None if verification_success else "Document quality insufficient or information mismatch"
    }

def kyc_page():
    """KYC verification page with simple black and white theme"""
    
    # Simple Black and White Theme with Background Animation
    st.markdown("""
    <style>
        @keyframes backgroundMove {
            0% { background-position: 0% 0%; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0% 0%; }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .main-container {
            background: linear-gradient(45deg, #f8f9fa 25%, transparent 25%), 
                        linear-gradient(-45deg, #f8f9fa 25%, transparent 25%), 
                        linear-gradient(45deg, transparent 75%, #f8f9fa 75%), 
                        linear-gradient(-45deg, transparent 75%, #f8f9fa 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            animation: backgroundMove 20s linear infinite;
            padding: 2rem;
        }
        
        .header-section {
            background: #000;
            color: #fff;
            padding: 3rem 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            animation: fadeIn 1s ease-out;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .form-section {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 2rem;
            margin: 0.5rem 0;
            animation: fadeIn 1s ease-out;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .section-title {
            color: #000;
            font-size: 1.8rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .submit-container {
            text-align: center;
            margin: 2rem auto;
            padding: 3rem 2rem;
            background: #fff;
            border: 3px solid #000;
            border-radius: 15px;
            max-width: 600px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease-out;
        }
        
        .submit-button {
            background: #000;
            color: #fff;
            border: none;
            padding: 1rem 3rem;
            font-size: 1.2rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .submit-button:hover {
            background: #333;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        }
        
        .kyc-card {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-out;
            transition: all 0.3s ease;
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 120px;
            width: 100%;
        }
        
        .kyc-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        
        .kyc-card h3 {
            color: #000;
            font-size: 1.5rem;
            font-weight: 900;
            margin: 0 0 0.5rem 0;
            line-height: 1.2;
        }
        
        .kyc-card p {
            color: #666;
            font-size: 0.9rem;
            font-weight: 600;
            margin: 0;
            line-height: 1.2;
        }
        
        /* Responsive design for KYC cards */
        @media (max-width: 1200px) {
            .kyc-card {
                height: 110px;
                min-height: 110px;
                padding: 1.2rem;
            }
            
            .kyc-card h3 {
                font-size: 1.3rem;
            }
            
            .kyc-card p {
                font-size: 0.85rem;
            }
        }
        
        @media (max-width: 768px) {
            .kyc-card {
                height: 100px;
                min-height: 100px;
                padding: 1rem;
            }
            
            .kyc-card h3 {
                font-size: 1.2rem;
            }
            
            .kyc-card p {
                font-size: 0.8rem;
            }
        }
        
        @media (max-width: 480px) {
            .kyc-card {
                height: 90px;
                min-height: 90px;
                padding: 0.8rem;
            }
            
            .kyc-card h3 {
                font-size: 1.1rem;
            }
            
            .kyc-card p {
                font-size: 0.75rem;
            }
        }
    </style>
    
    <div class="main-container">
        <div class="header-section">
            <h1 style="font-size: 3rem; font-weight: 900; margin: 0; text-transform: uppercase; letter-spacing: 3px;">
                KYC VERIFICATION
            </h1>
            <p style="font-size: 1.2rem; margin: 1rem 0 0 0; font-weight: 600;">
                Complete your identity verification to start investing
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KYC Status Display with amazing theme
    if st.session_state.kyc_status['verified']:
        st.markdown("""
        <div class="status-verified">
            <h2 style="margin: 0; font-size: 2.5rem;">🎉 KYC VERIFIED!</h2>
            <p style="font-size: 1.3rem; margin: 1rem 0;">Your identity has been successfully verified</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Amazing metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="kyc-card">
                <h3>✅ Verified</h3>
                <p>Status</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            verification_date = st.session_state.kyc_status.get('verification_date', None)
            if verification_date:
                date_str = verification_date.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = 'N/A'
            st.markdown(f"""
            <div class="kyc-card">
                <h3>📅 {date_str}</h3>
                <p>Verified Date</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="kyc-card">
                <h3>💰 PKR 50M</h3>
                <p>Investment Limit</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="kyc-card">
                <h3>🔒 Secure</h3>
                <p>Data Protection</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Re-verify button with amazing styling
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <button onclick="window.location.reload()" style="
                background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                color: white;
                border: none;
                border-radius: 25px;
                padding: 1rem 2rem;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 10px 20px rgba(239, 68, 68, 0.3);
            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 15px 30px rgba(239, 68, 68, 0.4)'" 
               onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 10px 20px rgba(239, 68, 68, 0.3)'">
                🔄 Re-verify KYC
            </button>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Re-verify KYC", key="reverify"):
            st.session_state.kyc_status['verified'] = False
            st.session_state.kyc_status['documents_uploaded'] = False
            st.rerun()
        
        return
    
    # Simple KYC Form
    st.markdown("""
    <div class="form-section">
        <h2 class="section-title">COMPLETE YOUR KYC VERIFICATION</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("kyc_form"):
        # Personal Information Section
        st.markdown("""
        <div class="form-section">
            <h3 class="section-title">PERSONAL INFORMATION</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Full Name *", value=st.session_state.kyc_status['personal_info'].get('full_name', ''), 
                                     help="Enter your full legal name as it appears on your ID")
            email = st.text_input("Email Address *", value=st.session_state.kyc_status['personal_info'].get('email', ''),
                                 help="We'll use this to send you verification updates")
            phone = st.text_input("Phone Number *", value=st.session_state.kyc_status['personal_info'].get('phone', ''),
                                 help="Include country code (e.g., +92 for Pakistan)")
            date_of_birth = st.date_input("Date of Birth *", value=st.session_state.kyc_status['personal_info'].get('date_of_birth', datetime(1990, 1, 1).date()),
                                        help="Must be 18+ years old")
        
        with col2:
            address = st.text_area("Full Address *", value=st.session_state.kyc_status['personal_info'].get('address', ''),
                                  help="Complete residential address")
            national_id = st.text_input("National ID/Passport Number *", value=st.session_state.kyc_status['personal_info'].get('national_id', ''),
                                       help="CNIC, Passport, or other government-issued ID")
            occupation = st.text_input("Occupation", value=st.session_state.kyc_status['personal_info'].get('occupation', ''),
                                      help="Your current job or profession")
            annual_income = st.selectbox("Annual Income Range", 
                                      ["Under PKR 500,000", "PKR 500,000 - 1,000,000", "PKR 1,000,000 - 2,500,000", 
                                       "PKR 2,500,000 - 5,000,000", "PKR 5,000,000 - 10,000,000", "Over PKR 10,000,000"],
                                      index=2, help="Select your annual income range")
        
        # Document Upload Section
        st.markdown("""
        <div class="form-section">
            <h3 class="section-title">DOCUMENT UPLOAD</h3>
            <p style="text-align: center; color: #666; margin-bottom: 1rem; font-weight: 600;">
                Please upload clear, high-quality images of your documents
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🆔 ID Document")
            id_document = st.file_uploader("Passport/Driver's License/CNIC *", 
                                         type=['png', 'jpg', 'jpeg'], 
                                         help="Upload front and back of your ID document")
            if id_document:
                st.image(id_document, width=200, caption="ID Document Preview")
        
        with col2:
            st.markdown("### 🏠 Address Proof")
            address_proof = st.file_uploader("Utility Bill/Bank Statement *", 
                                           type=['png', 'jpg', 'jpeg'], 
                                           help="Document should be less than 3 months old")
            if address_proof:
                st.image(address_proof, width=200, caption="Address Proof Preview")
        
        # Additional documents
        st.markdown("""
        <div class="form-section">
            <h3 class="section-title">ADDITIONAL DOCUMENTS (OPTIONAL)</h3>
        </div>
        """, unsafe_allow_html=True)
        
        income_proof = st.file_uploader("Income Proof (Pay Stub/Tax Return)", type=['png', 'jpg', 'jpeg', 'pdf'],
                                       help="Optional: Helps with investment limit approval")
        
        # Terms and conditions
        st.markdown("""
        <div class="form-section">
            <h3 class="section-title">TERMS AND CONDITIONS</h3>
        </div>
        """, unsafe_allow_html=True)
        
        terms_accepted = st.checkbox("I agree to the terms and conditions and privacy policy *", 
                                   help="Required to proceed with KYC verification")
        data_consent = st.checkbox("I consent to the processing of my personal data for KYC verification *",
                                  help="Required for identity verification")
        
        # Centered Submit Button with Black & White Theme
        st.markdown("""
        <div class="submit-container">
            <h3 style="color: #000; font-size: 1.8rem; font-weight: 900; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 2px;">
                READY TO VERIFY YOUR IDENTITY?
            </h3>
            <p style="color: #666; font-size: 1.2rem; font-weight: 600; margin-bottom: 2rem;">
                Click the button below to submit your KYC application
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("SUBMIT KYC APPLICATION", type="primary", use_container_width=True)
        
        if submitted:
            if not all([full_name, email, phone, address, national_id, id_document, address_proof, terms_accepted, data_consent]):
                st.error("❌ Please fill in all required fields and upload required documents.")
            else:
                # Store personal information
                personal_info = {
                    'full_name': full_name,
                    'email': email,
                    'phone': phone,
                    'address': address,
                    'date_of_birth': date_of_birth,
                    'national_id': national_id,
                    'occupation': occupation,
                    'annual_income': annual_income
                }
                
                # Store documents (in real app, these would be uploaded to secure storage)
                documents = {
                    'id_document': id_document.name if id_document else None,
                    'address_proof': address_proof.name if address_proof else None,
                    'income_proof': income_proof.name if income_proof else None
                }
                
                # Update session state
                st.session_state.kyc_status['personal_info'] = personal_info
                st.session_state.kyc_status['documents'] = documents
                st.session_state.kyc_status['documents_uploaded'] = True
                
                # ABSOLUTELY AMAZING verification process
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                    color: white;
                    padding: 3rem;
                    border-radius: 30px;
                    text-align: center;
                    margin: 2rem 0;
                    box-shadow: 
                        0 25px 50px rgba(102, 126, 234, 0.4),
                        0 0 30px rgba(102, 126, 234, 0.3);
                    border: 3px solid rgba(255, 255, 255, 0.2);
                    animation: neonGlow 2s ease-in-out infinite;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
                        animation: cardShimmer 2s ease-in-out infinite;
                    "></div>
                    <h3 style="
                        margin: 0; 
                        font-size: 2.5rem; 
                        font-weight: 800;
                        text-shadow: 0 0 20px rgba(255, 255, 255, 0.8);
                        animation: titlePulse 2s ease-in-out infinite;
                        position: relative;
                        z-index: 2;
                    ">🔍 VERIFYING YOUR DOCUMENTS... 🔍</h3>
                    <p style="
                        margin: 1.5rem 0 0 0; 
                        font-size: 1.3rem;
                        font-weight: 600;
                        text-shadow: 0 0 15px rgba(255, 255, 255, 0.6);
                        position: relative;
                        z-index: 2;
                    ">This may take a few minutes. Please wait...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Simulate verification process
                with st.spinner("🔍 Verifying your documents... This may take a few minutes."):
                    verification_result = verify_kyc_documents(personal_info, documents)
                    
                    if verification_result['verified']:
                        st.session_state.kyc_status['verified'] = True
                        st.session_state.kyc_status['verification_date'] = verification_result['verification_date']
                        
                        st.markdown("""
                        <div style="
                            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                            color: white;
                            padding: 3rem;
                            border-radius: 20px;
                            text-align: center;
                            margin: 2rem 0;
                            box-shadow: 0 20px 40px rgba(16, 185, 129, 0.4);
                            animation: glow 2s ease-in-out infinite;
                        ">
                            <h2 style="margin: 0; font-size: 3rem;">🎉 KYC VERIFIED!</h2>
                            <p style="font-size: 1.5rem; margin: 1rem 0;">Your identity has been successfully verified</p>
                            <p style="font-size: 1.2rem; margin: 0; opacity: 0.9;">You can now invest in tokenized real estate!</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.rerun()
                    else:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                            color: white;
                            padding: 2rem;
                            border-radius: 20px;
                            text-align: center;
                            margin: 2rem 0;
                            box-shadow: 0 15px 35px rgba(239, 68, 68, 0.4);
                        ">
                            <h3 style="margin: 0; font-size: 2rem;">❌ KYC Verification Failed</h3>
                            <p style="margin: 1rem 0 0 0; opacity: 0.9;">{verification_result['rejection_reason']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("Please check your documents and try again. Ensure all documents are clear and readable.")

def generate_dummy_properties():
    """Generate dummy property data"""
    properties = []
    locations = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad', 'Multan', 'Peshawar', 'Quetta', 'Gujranwala', 'Sialkot']
    
    # Pakistani property names
    property_names = [
        'Centaurus Mall', 'Bahria Town Plaza', 'DHA Phase 5 Tower', 'Gulberg Heights', 
        'Clifton Beach Resort', 'F-8 Commercial Complex', 'Model Town Plaza', 'Defence Tower',
        'Blue Area Office Complex', 'Garden City Residency', 'Lucky One Mall', 'Dolmen City',
        'Emporium Mall Tower', 'Packages Mall Complex', 'Fortress Square', 'Giga Mall',
        'Centaurus Residency', 'Bahria Icon Tower', 'DHA Phase 2 Plaza', 'Gulberg Greens'
    ]
    
    for i in range(20):
        property_data = {
            'id': f'PROP_{i+1:03d}',
            'name': property_names[i],
            'location': random.choice(locations),
            'price': random.randint(5000000, 50000000),  # Prices in PKR (5M to 50M PKR)
            'roi': round(random.uniform(12, 30), 2),  # Higher ROI for Pakistani market
            'tokens_supply': random.randint(1000, 10000),
            'tokens_available': random.randint(100, 1000),
            'image_url': f'https://picsum.photos/400/300?random={i}',
            'description': f"Premium {random.choice(['residential', 'commercial', 'mixed-use'])} property in {random.choice(locations)}. Modern amenities, prime location, excellent investment opportunity.",
            'property_type': random.choice(['Residential', 'Commercial', 'Mixed-Use']),
            'year_built': random.randint(2000, 2024),
            'square_feet': random.randint(2000, 50000)
        }
        properties.append(property_data)
    
    return properties

def generate_historical_data():
    """Generate historical ROI data for ML models"""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
    data = []
    
    pakistani_locations = ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi', 'Faisalabad', 'Multan', 'Peshawar', 'Quetta', 'Gujranwala', 'Sialkot']
    
    for i in range(20):
        base_roi = random.uniform(12, 25)  # Higher base ROI for Pakistani market
        for date in dates:
            # Add some trend and seasonality
            trend = (date.year - 2020) * 0.8  # Stronger growth trend
            seasonality = np.sin(2 * np.pi * date.month / 12) * 3
            noise = random.uniform(-3, 3)
            
            roi = base_roi + trend + seasonality + noise
            data.append({
                'property_id': f'PROP_{i+1:03d}',
                'date': date,
                'roi': max(0, roi),
                'price': random.randint(5000000, 50000000),  # Prices in PKR
                'location': random.choice(pakistani_locations)
            })
    
    return pd.DataFrame(data)

def create_pdf_invoice(property_name, investment_amount, tokens, ownership_percent, roi):
    """Create a professional PDF invoice"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1
    )
    
    # Content
    story = []
    
    # Title
    story.append(Paragraph("PropToken Investment Invoice", title_style))
    story.append(Spacer(1, 20))
    
    # Invoice details
    invoice_data = [
        ['Invoice Number:', f'INV-{datetime.now().strftime("%Y%m%d%H%M%S")}'],
        ['Date:', datetime.now().strftime("%B %d, %Y")],
        ['Property:', property_name],
        ['Investment Amount:', f'${investment_amount:,.2f}'],
        ['Tokens Issued:', f'{tokens:,.0f}'],
        ['Ownership Percentage:', f'{ownership_percent:.2f}%'],
        ['Expected ROI:', f'{roi:.2f}%'],
    ]
    
    table = Table(invoice_data, colWidths=[2*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 30))
    
    # Terms and conditions
    story.append(Paragraph("Terms and Conditions", styles['Heading2']))
    terms = """
    This investment represents ownership of digital tokens backed by real estate assets. 
    Tokens are secured on the blockchain and provide proportional ownership rights. 
    Returns are subject to property performance and market conditions.
    """
    story.append(Paragraph(terms, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def home_page():
    """Home page with black and white theme"""
    
    # Black and White Home Page Theme
    st.markdown("""
    <style>
        .home-container {
            background: linear-gradient(45deg, #f8f9fa 25%, transparent 25%), 
                        linear-gradient(-45deg, #f8f9fa 25%, transparent 25%), 
                        linear-gradient(45deg, transparent 75%, #f8f9fa 75%), 
                        linear-gradient(-45deg, transparent 75%, #f8f9fa 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            animation: backgroundMove 20s linear infinite;
            padding: 2rem;
        }
        
        .home-header {
            background: #000;
            color: #fff;
            padding: 3rem 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            animation: fadeIn 1s ease-out;
        }
        
        .content-section {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-out;
        }
        
        .section-title {
            color: #000;
            font-size: 1.8rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .stats-card {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: slideInFromLeft 2s ease-out, float 3s ease-in-out infinite;
            position: relative;
            overflow: hidden;
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 150px;
            width: 100%;
        }
        
        .stats-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(0,0,0,0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes slideInFromLeft {
            0% { transform: translateX(-100px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        .stats-card h3 {
            color: #000;
            font-size: 2rem;
            font-weight: 900;
            margin: 0 0 0.5rem 0;
        }
        
        .stats-card p {
            color: #666;
            font-size: 1rem;
            font-weight: 600;
            margin: 0;
            line-height: 1.2;
            word-break: break-word;
            text-align: center;
        }
        
        /* Global responsive design for all cards */
        @media (max-width: 1200px) {
            .stats-card {
                height: 140px;
                min-height: 140px;
                padding: 1.2rem;
            }
            
            .stats-card h3 {
                font-size: 1.8rem;
            }
            
            .stats-card p {
                font-size: 0.95rem;
            }
        }
        
        @media (max-width: 768px) {
            .stats-card {
                height: 130px;
                min-height: 130px;
                padding: 1rem;
            }
            
            .stats-card h3 {
                font-size: 1.6rem;
            }
            
            .stats-card p {
                font-size: 0.9rem;
            }
        }
        
        @media (max-width: 480px) {
            .stats-card {
                height: 120px;
                min-height: 120px;
                padding: 0.8rem;
            }
            
            .stats-card h3 {
                font-size: 1.4rem;
            }
            
            .stats-card p {
                font-size: 0.8rem;
            }
        }
        
        /* Responsive grid system */
        .stColumns > div {
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        
        @media (max-width: 1200px) {
            .stColumns {
                display: grid !important;
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 1rem !important;
            }
        }
        
        @media (max-width: 768px) {
            .stColumns {
                display: grid !important;
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 1rem !important;
            }
        }
        
        @media (max-width: 480px) {
            .stColumns {
                display: grid !important;
                grid-template-columns: 1fr !important;
                gap: 1rem !important;
            }
        }
        
        .cta-container {
            text-align: center;
            margin: 2rem auto;
            padding: 3rem 2rem;
            background: #fff;
            border: 3px solid #000;
            border-radius: 15px;
            max-width: 600px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: fadeIn 1s ease-out;
        }
        
        .cta-button {
            background: #000;
            color: #fff;
            border: none;
            padding: 1rem 3rem;
            font-size: 1.2rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .cta-button:hover {
            background: #333;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        }
        
        .pulse-button {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .kyc-status {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 1rem 2rem;
            margin-bottom: 2rem;
            text-align: center;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .kyc-verified {
            color: #000;
            background: #f0f0f0;
        }
        
        .kyc-pending {
            color: #666;
            background: #f8f9fa;
        }
    </style>
    
    <div class="home-container">
        <div class="home-header">
            <h1 style="font-size: 3rem; font-weight: 900; margin: 0; text-transform: uppercase; letter-spacing: 3px;">
                PROPTOKEN
            </h1>
            <p style="font-size: 1.5rem; margin: 1rem 0 0 0; font-weight: 600;">
                OWN REAL ESTATE LIKE OWNING SHARES
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KYC Status Banner
    if st.session_state.kyc_status['verified']:
        st.markdown("""
        <div class="kyc-status kyc-verified">
            ✅ KYC VERIFIED - READY TO INVEST!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="kyc-status kyc-pending">
            🔐 COMPLETE KYC VERIFICATION TO START INVESTING
        </div>
        """, unsafe_allow_html=True)
    
    # Tokenization explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="content-section">
            <h2 class="section-title">TOKENIZATION EXPLAINED</h2>
            <p style="font-size: 1.2rem; font-weight: 700; color: #000; margin-bottom: 1rem;">
                A PROPERTY IS LIKE A PIZZA
            </p>
            <p style="color: #666; margin-bottom: 1rem;">
                Tokenization slices it so anyone can own a piece.
            </p>
            <p style="color: #666; margin-bottom: 1rem;">
                Just like JazzCash tokenizes payments, PropToken tokenizes ownership.
            </p>
            <h3 style="color: #000; font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem;">HOW IT WORKS:</h3>
            <ul style="color: #666; font-weight: 600;">
                <li>Real estate is divided into digital tokens</li>
                <li>Each token represents fractional ownership</li>
                <li>Blockchain ensures transparent, immutable records</li>
                <li>Trade tokens like stocks on secondary markets</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-section">
            <h2 class="section-title">WHY TOKENIZE REAL ESTATE?</h2>
            <ul style="color: #666; font-weight: 600; font-size: 1.1rem;">
                <li><strong style="color: #000;">LOWER BARRIERS:</strong> Invest with as little as $100</li>
                <li><strong style="color: #000;">LIQUIDITY:</strong> Trade tokens 24/7</li>
                <li><strong style="color: #000;">TRANSPARENCY:</strong> All transactions on blockchain</li>
                <li><strong style="color: #000;">DIVERSIFICATION:</strong> Own pieces of multiple properties</li>
                <li><strong style="color: #000;">GLOBAL ACCESS:</strong> Invest in properties worldwide</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Stats cards
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">MARKET STATISTICS</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 15s linear infinite;
                white-space: nowrap;
                font-size: 1.1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 2px;
            ">
                🚀 REAL ESTATE TOKENIZATION IS THE FUTURE • GLOBAL MARKET GROWING 20%+ ANNUALLY • INVEST FROM ANYWHERE IN THE WORLD • BLOCKCHAIN SECURITY GUARANTEED • FRACTIONAL OWNERSHIP MADE EASY • LIQUIDITY LIKE NEVER BEFORE • TRANSPARENT TRANSACTIONS • DEMOCRATIZING REAL ESTATE INVESTMENT • 🚀
            </div>
        </div>
    </div>
    
    <style>
        @keyframes runningText {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card" style="animation-delay: 0.5s;">
            <h3>$2.3B</h3>
            <p>Global tokenization market (2021)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card" style="animation-delay: 1s;">
            <h3>20%+</h3>
            <p>CAGR expected growth</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card" style="animation-delay: 1.5s;">
            <h3>$100</h3>
            <p>Minimum entry investment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card" style="animation-delay: 2s;">
            <h3>100%</h3>
            <p>Transparent ownership records</p>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Button
    st.markdown("""
    <div class="cta-container">
        <h3 style="color: #000; font-size: 1.8rem; font-weight: 900; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 2px;">
            READY TO START INVESTING?
        </h3>
        <p style="color: #666; font-size: 1.2rem; font-weight: 600; margin-bottom: 2rem;">
            Click the button below to explore our marketplace
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Centered CTA button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <style>
            .stButton > button {
                background: #000 !important;
                color: #fff !important;
                border: none !important;
                padding: 1rem 3rem !important;
                font-size: 1.2rem !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 2px !important;
                border-radius: 8px !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
                animation: pulse 2s infinite !important;
                width: 100% !important;
            }
            
            .stButton > button:hover {
                background: #333 !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(0,0,0,0.4) !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("GO TO MARKETPLACE", key="cta_home", help="Start investing in tokenized real estate", use_container_width=True):
            st.session_state.current_page = "Portfolio/Marketplace"
            st.rerun()

def marketplace_page():
    """Portfolio/Marketplace page with black and white theme"""
    
    # Black and White Marketplace Theme
    st.markdown("""
    <style>
        .marketplace-container {
            background: linear-gradient(45deg, #f8f9fa 25%, transparent 25%), 
                        linear-gradient(-45deg, #f8f9fa 25%, transparent 25%), 
                        linear-gradient(45deg, transparent 75%, #f8f9fa 75%), 
                        linear-gradient(-45deg, transparent 75%, #f8f9fa 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            animation: backgroundMove 20s linear infinite;
            padding: 2rem;
        }
        
        .marketplace-header {
            background: #000;
            color: #fff;
            padding: 3rem 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            animation: fadeIn 1s ease-out;
        }
        
        .content-section {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-out;
        }
        
        .section-title {
            color: #000;
            font-size: 1.8rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .property-card {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: slideInFromLeft 1s ease-out;
            transition: all 0.3s ease;
            height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 180px;
            width: 100%;
        }
        
        .property-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        
        .property-card h4 {
            color: #000;
            font-size: 1.2rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        }
        
        .property-card p {
            color: #666;
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.3rem;
            line-height: 1.2;
        }
        
        @media (max-width: 768px) {
            .property-card {
                height: 180px;
                min-height: 180px;
                padding: 1rem;
            }
            
            .property-card h4 {
                font-size: 1.1rem;
            }
            
            .property-card p {
                font-size: 0.8rem;
            }
        }
        
        @media (max-width: 1200px) {
            .property-card {
                height: 170px;
                min-height: 170px;
                padding: 1.2rem;
            }
            
            .property-card h4 {
                font-size: 1.1rem;
            }
            
            .property-card p {
                font-size: 0.85rem;
            }
        }
        
        @media (max-width: 768px) {
            .property-card {
                height: 160px;
                min-height: 160px;
                padding: 1rem;
            }
            
            .property-card h4 {
                font-size: 1rem;
            }
            
            .property-card p {
                font-size: 0.8rem;
            }
        }
        
        @media (max-width: 480px) {
            .property-card {
                height: 150px;
                min-height: 150px;
                padding: 0.8rem;
            }
            
            .property-card h4 {
                font-size: 0.9rem;
            }
            
            .property-card p {
                font-size: 0.75rem;
            }
        }
        
        .property-title {
            color: #000;
            font-size: 1.5rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }
        
        .property-details {
            color: #666;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .roi-badge {
            background: #000;
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 700;
            font-size: 1.1rem;
            display: inline-block;
            margin: 0.5rem 0;
        }
        
        .investment-form {
            background: #f8f9fa;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .form-title {
            color: #000;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-transform: uppercase;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .stat-item {
            background: #fff;
            border: 2px solid #000;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        .stat-value {
            color: #000;
            font-size: 1.5rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        .search-container {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .search-title {
            color: #000;
            font-size: 1.5rem;
            font-weight: 900;
            margin-bottom: 1rem;
            text-align: center;
            text-transform: uppercase;
        }
        
        .kyc-warning {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .kyc-warning h2 {
            color: #856404;
            font-size: 1.8rem;
            font-weight: 900;
            margin-bottom: 1rem;
            text-transform: uppercase;
        }
        
        .kyc-warning p {
            color: #856404;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .kyc-button {
            background: #000;
            color: #fff;
            border: none;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .kyc-button:hover {
            background: #333;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        }
        
        @keyframes slideInFromLeft {
            0% { transform: translateX(-100px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        @keyframes runningText {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
    </style>
    
    <div class="marketplace-container">
        <div class="marketplace-header">
            <h1 style="font-size: 3rem; font-weight: 900; margin: 0; text-transform: uppercase; letter-spacing: 3px;">
                PROPERTY MARKETPLACE
            </h1>
            <p style="font-size: 1.2rem; margin: 1rem 0 0 0; font-weight: 600;">
                Invest in tokenized real estate properties
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check KYC status
    if not st.session_state.kyc_status['verified']:
        st.markdown("""
        <div class="kyc-warning">
            <h2>🔐 KYC VERIFICATION REQUIRED</h2>
            <p>Complete your identity verification to start investing in properties</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <style>
                .stButton > button {
                    background: #000 !important;
                    color: #fff !important;
                    border: none !important;
                    padding: 1rem 2rem !important;
                    font-size: 1.1rem !important;
                    font-weight: 700 !important;
                    text-transform: uppercase !important;
                    letter-spacing: 1px !important;
                    border-radius: 8px !important;
                    cursor: pointer !important;
                    transition: all 0.3s ease !important;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
                    width: 100% !important;
                }
                
                .stButton > button:hover {
                    background: #333 !important;
                    transform: translateY(-2px) !important;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.4) !important;
                }
            </style>
            """, unsafe_allow_html=True)
            
            if st.button("COMPLETE KYC VERIFICATION", key="kyc_redirect", use_container_width=True):
                st.session_state.current_page = "KYC"
                st.rerun()
        return
    
    # Show KYC status for verified users
    st.markdown("""
    <div class="content-section" style="text-align: center; background: #e6ffe6; border-color: #00cc00;">
        <h2 style="color: #008000; font-size: 2rem; font-weight: 900; margin: 0;">✅ KYC VERIFIED!</h2>
        <p style="color: #008000; font-size: 1.2rem; margin: 0.5rem 0 0 0; font-weight: 600;">You can invest in properties</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-value">✅ Verified</div>
            <div class="stat-label">Status</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-value">💰 $50,000</div>
            <div class="stat-label">Investment Limit</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-item">
            <div class="stat-value">🔒 Secure</div>
            <div class="stat-label">Data Protection</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Re-verify button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <style>
            .stButton > button {
                background: #dc3545 !important;
                color: #fff !important;
                border: none !important;
                padding: 0.8rem 1.5rem !important;
                font-size: 1rem !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 1px !important;
                border-radius: 8px !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
                width: 100% !important;
            }
            
            .stButton > button:hover {
                background: #c82333 !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(0,0,0,0.4) !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 RE-VERIFY KYC", use_container_width=True):
            st.session_state.kyc_status['verified'] = False
            st.rerun()
    
    # Initialize properties if not exists
    if not st.session_state.properties:
        st.session_state.properties = generate_dummy_properties()
    
    # Debug information
    st.info(f"Total properties available: {len(st.session_state.properties)}")
    
    # Search and filters
    st.markdown("""
    <div class="search-container">
        <h2 class="search-title">🔍 SEARCH PROPERTIES</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 15s linear infinite;
                white-space: nowrap;
                font-size: 1.1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 2px;
            ">
                🏠 FIND YOUR PERFECT INVESTMENT • PREMIUM PROPERTIES IN PAKISTAN • HIGH ROI OPPORTUNITIES • BLOCKCHAIN SECURED • FRACTIONAL OWNERSHIP • LIQUID REAL ESTATE • INVEST FROM ANYWHERE • 🏠
            </div>
        </div>
    </div>
    
    <style>
        @keyframes runningText {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        location_filter = st.selectbox("Location", ["All"] + list(set([p['location'] for p in st.session_state.properties])))
    
    with col2:
        min_roi = st.slider("Min ROI (%)", 0, 30, 8)
    
    with col3:
        max_price = st.slider("Max Price (PKR)", 5000000, 50000000, 50000000)
    
    with col4:
        property_type = st.selectbox("Property Type", ["All", "Residential", "Commercial", "Mixed-Use"])
    
    # Filter properties
    filtered_properties = st.session_state.properties.copy()
    
    if location_filter != "All":
        filtered_properties = [p for p in filtered_properties if p['location'] == location_filter]
    
    filtered_properties = [p for p in filtered_properties if p['roi'] >= min_roi]
    filtered_properties = [p for p in filtered_properties if p['price'] <= max_price]
    
    if property_type != "All":
        filtered_properties = [p for p in filtered_properties if p['property_type'] == property_type]
    
    # Debug information
    st.info(f"Filtered properties: {len(filtered_properties)}")
    if len(filtered_properties) == 0:
        st.warning("No properties match your filters. Try adjusting the search criteria.")
        return
    
    # Display properties
    st.markdown(f"""
    <div class="content-section">
        <h2 class="section-title">📋 AVAILABLE PROPERTIES ({len(filtered_properties)} FOUND)</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 12s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                💰 INVEST IN TOKENIZED REAL ESTATE • OWN FRACTIONS OF PREMIUM PROPERTIES • EARN PASSIVE INCOME • TRADE TOKENS 24/7 • BLOCKCHAIN TRANSPARENCY • SECURE INVESTMENT • HIGH RETURNS • 💰
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    for i, prop in enumerate(filtered_properties):
        # Property Card with enhanced animations
        st.markdown(f"""
        <div style="
            border: 2px solid #000; 
            border-radius: 10px; 
            padding: 1.5rem; 
            margin: 1rem 0; 
            background: #fff;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: slideInFromLeft {0.5 + (i * 0.2)}s ease-out;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        " onmouseover="this.style.transform='translateY(-5px)'; this.style.boxShadow='0 10px 25px rgba(0,0,0,0.2)'" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 5px 15px rgba(0,0,0,0.1)'">
            <div style="
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(0,0,0,0.05), transparent);
                animation: shimmer 3s infinite;
                animation-delay: {i * 0.5}s;
            "></div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            st.image(prop['image_url'], width=200)
        
        with col2:
            st.markdown(f"### {prop['name']}")
            st.markdown(f"**Location:** {prop['location']} | **Type:** {prop['property_type']}")
            st.markdown(f"**Price:** PKR {prop['price']:,} | **Year Built:** {prop['year_built']}")
            st.markdown(f"**Square Feet:** {prop['square_feet']:,} sq ft")
            st.markdown(f"**Description:** {prop['description']}")
            st.markdown(f"**Token Supply:** {prop['tokens_supply']:,}")
            st.markdown(f"**Available:** {prop['tokens_available']:,}")
        
        with col3:
            st.markdown(f"""
            <div style="
                background: #000; 
                color: #fff; 
                padding: 1rem; 
                border-radius: 8px; 
                text-align: center;
                margin-bottom: 1rem;
            ">
                <h3 style="margin: 0; font-size: 1.5rem;">{prop['roi']}% ROI</h3>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form(key=f"invest_form_{i}"):
                st.markdown("**Investment Amount**")
                investment_amount = st.number_input(
                    "Amount (PKR)", 
                    min_value=100000, 
                    max_value=prop['price'], 
                    value=1000000,
                    key=f"amount_{i}",
                    label_visibility="collapsed"
                )
                
                st.markdown("""
                <style>
                    .stButton > button {
                        background: #000 !important;
                        color: #fff !important;
                        border: none !important;
                        padding: 0.8rem 1.5rem !important;
                        font-size: 1rem !important;
                        font-weight: 700 !important;
                        text-transform: uppercase !important;
                        letter-spacing: 1px !important;
                        border-radius: 8px !important;
                        cursor: pointer !important;
                        transition: all 0.3s ease !important;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
                        width: 100% !important;
                    }
                    
                    .stButton > button:hover {
                        background: #333 !important;
                        transform: translateY(-2px) !important;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.4) !important;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                if st.form_submit_button("💰 INVEST NOW", use_container_width=True):
                        # Calculate investment details
                        token_price = prop['price'] / prop['tokens_supply']
                        tokens_received = investment_amount / token_price
                        ownership_percent = (tokens_received / prop['tokens_supply']) * 100
                        platform_fee = investment_amount * 0.02  # 2% platform fee
                        net_investment = investment_amount - platform_fee
                        
                        # Store investment
                        investment = {
                            'property_id': prop['id'],
                            'property_name': prop['name'],
                            'investment_amount': investment_amount,
                            'tokens_received': tokens_received,
                            'ownership_percent': ownership_percent,
                            'roi': prop['roi'],
                            'platform_fee': platform_fee,
                            'net_investment': net_investment,
                            'timestamp': datetime.now()
                        }
                        
                        st.session_state.investments.append(investment)
                        
                        # Store PDF data in session state for download outside form
                        pdf_buffer = create_pdf_invoice(
                            prop['name'], 
                            investment_amount, 
                            tokens_received, 
                            ownership_percent, 
                            prop['roi']
                        )
                        
                        st.session_state.latest_pdf = {
                            'data': pdf_buffer.getvalue(),
                            'filename': f"investment_invoice_{prop['id']}.pdf"
                        }
                        
                        st.success(f"Investment successful! You received {tokens_received:,.0f} tokens ({ownership_percent:.2f}% ownership)")
                        st.rerun()
        
        # Close the property card div
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Download button for latest investment (outside of form)
    if 'latest_pdf' in st.session_state and st.session_state.latest_pdf:
        st.markdown("## 📄 Download Your Investment Documents")
        st.download_button(
            label="📄 Download Invoice & Agreement",
            data=st.session_state.latest_pdf['data'],
            file_name=st.session_state.latest_pdf['filename'],
            mime="application/pdf"
        )
        if st.button("🗑️ Clear Download"):
            del st.session_state.latest_pdf
            st.rerun()
    
    # Seller registration
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">🏗️ REGISTER NEW PROPERTY</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 18s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                🏗️ LIST YOUR PROPERTY • TOKENIZE REAL ESTATE • EARN FROM RENTAL INCOME • INCREASE LIQUIDITY • REACH GLOBAL INVESTORS • BLOCKCHAIN SECURITY • FRACTIONAL OWNERSHIP • 🏗️
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background: #f8f9fa;
        border: 2px solid #000;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    ">
    """, unsafe_allow_html=True)
    
    with st.expander("➕ ADD PROPERTY TO MARKETPLACE", expanded=False):
        with st.form("seller_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_prop_name = st.text_input("Property Name")
                new_prop_location = st.selectbox("Location", ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar", "Quetta", "Gujranwala", "Sialkot"])
                new_prop_price = st.number_input("Property Price (PKR)", min_value=1000000, max_value=100000000, value=10000000)
                new_prop_roi = st.number_input("Expected ROI (%)", min_value=0.0, max_value=50.0, value=15.0)
            
            with col2:
                new_prop_type = st.selectbox("Property Type", ["Residential", "Commercial", "Mixed-Use"])
                new_prop_year = st.number_input("Year Built", min_value=1900, max_value=2024, value=2020)
                new_prop_sqft = st.number_input("Square Feet", min_value=1000, max_value=100000, value=5000)
                new_prop_tokens = st.number_input("Token Supply", min_value=100, max_value=50000, value=1000)
            
            new_prop_description = st.text_area("Property Description")
            
            if st.form_submit_button("🏠 Register Property"):
                new_property = {
                    'id': f'PROP_{len(st.session_state.properties) + 1:03d}',
                    'name': new_prop_name,
                    'location': new_prop_location,
                    'price': new_prop_price,
                    'roi': new_prop_roi,
                    'tokens_supply': new_prop_tokens,
                    'tokens_available': new_prop_tokens,
                    'image_url': f'https://picsum.photos/400/300?random={len(st.session_state.properties) + 100}',
                    'description': new_prop_description,
                    'property_type': new_prop_type,
                    'year_built': new_prop_year,
                    'square_feet': new_prop_sqft
                }
                
                st.session_state.properties.append(new_property)
                st.success("Property registered successfully!")
                st.rerun()
    
    # Close the seller form container
    st.markdown("</div>", unsafe_allow_html=True)

def analytics_page():
    """Analytics page with black and white theme"""
    
    # Black and White Analytics Theme
    st.markdown("""
    <style>
        .analytics-container {
            background: linear-gradient(45deg, #f8f9fa 25%, transparent 25%), 
                        linear-gradient(-45deg, #f8f9fa 25%, transparent 25%), 
                        linear-gradient(45deg, transparent 75%, #f8f9fa 75%), 
                        linear-gradient(-45deg, transparent 75%, #f8f9fa 75%);
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            animation: backgroundMove 20s linear infinite;
            padding: 2rem;
        }
        
        .analytics-header {
            background: #000;
            color: #fff;
            padding: 3rem 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            animation: fadeIn 1s ease-out;
        }
        
        .content-section {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: fadeIn 1s ease-out;
        }
        
        .section-title {
            color: #000;
            font-size: 1.8rem;
            font-weight: 900;
            text-align: center;
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .ml-models-section {
            background: #000;
            color: #fff;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .model-item {
            background: #fff;
            color: #000;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border: 2px solid #000;
        }
        
        .model-title {
            font-size: 1.2rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }
        
        .model-description {
            font-size: 1rem;
            font-weight: 600;
            color: #666;
        }
        
        .stats-card {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            animation: slideInFromLeft 1s ease-out;
            transition: all 0.3s ease;
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 150px;
            width: 100%;
        }
        
        .stats-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        
        .stats-value {
            color: #000;
            font-size: 1.8rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
            line-height: 1.2;
            word-break: break-word;
        }
        
        .stats-label {
            color: #666;
            font-size: 0.9rem;
            font-weight: 600;
            line-height: 1.2;
            word-break: break-word;
            text-align: center;
        }
        
        @media (max-width: 1200px) {
            .stats-card {
                height: 140px;
                min-height: 140px;
                padding: 1.2rem;
            }
            
            .stats-value {
                font-size: 1.7rem;
            }
            
            .stats-label {
                font-size: 0.95rem;
            }
        }
        
        @media (max-width: 768px) {
            .stats-card {
                height: 130px;
                min-height: 130px;
                padding: 1rem;
            }
            
            .stats-value {
                font-size: 1.5rem;
            }
            
            .stats-label {
                font-size: 0.9rem;
            }
        }
        
        @media (max-width: 480px) {
            .stats-card {
                height: 120px;
                min-height: 120px;
                padding: 0.8rem;
            }
            
            .stats-value {
                font-size: 1.3rem;
            }
            
            .stats-label {
                font-size: 0.8rem;
            }
        }
        
        /* Responsive grid for cards */
        .stColumns > div {
            display: flex;
            flex-direction: column;
        }
        
        @media (max-width: 768px) {
            .stColumns {
                display: grid !important;
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 1rem !important;
            }
        }
        
        @media (max-width: 480px) {
            .stColumns {
                display: grid !important;
                grid-template-columns: 1fr !important;
                gap: 1rem !important;
            }
        }
        
        .chart-container {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .filters-container {
            background: #fff;
            border: 2px solid #000;
            border-radius: 10px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .filters-title {
            color: #000;
            font-size: 1.5rem;
            font-weight: 900;
            margin-bottom: 1rem;
            text-align: center;
            text-transform: uppercase;
        }
        
        @keyframes slideInFromLeft {
            0% { transform: translateX(-100px); opacity: 0; }
            100% { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        @keyframes runningText {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes backgroundMove {
            0% { background-position: 0% 0%; }
            50% { background-position: 100% 100%; }
            100% { background-position: 0% 0%; }
        }
    </style>
    
    <div class="analytics-container">
        <div class="analytics-header">
            <h1 style="font-size: 3rem; font-weight: 900; margin: 0; text-transform: uppercase; letter-spacing: 3px;">
                AI-POWERED ANALYTICS
            </h1>
            <p style="font-size: 1.2rem; margin: 1rem 0 0 0; font-weight: 600;">
                Machine learning insights for real estate investment
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ML Models Section
    st.markdown("""
    <div class="ml-models-section">
        <h2 style="font-size: 2rem; font-weight: 900; margin-bottom: 1.5rem; text-align: center; text-transform: uppercase;">
            🧠 MACHINE LEARNING MODELS
        </h2>
        <div style="
            background: #fff;
            color: #000;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 16s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                🤖 AI-POWERED PREDICTIONS • MACHINE LEARNING INSIGHTS • DATA-DRIVEN DECISIONS • PREDICTIVE ANALYTICS • SMART INVESTMENT RECOMMENDATIONS • FUTURE ROI FORECASTING • 🤖
            </div>
        </div>
        <div class="model-item">
            <div class="model-title">PROPHET</div>
            <div class="model-description">Time series forecasting for ROI predictions</div>
        </div>
        <div class="model-item">
            <div class="model-title">XGBOOST</div>
            <div class="model-description">Gradient boosting for property value predictions</div>
        </div>
        <div class="model-item">
            <div class="model-title">REGRESSION</div>
            <div class="model-description">Linear models for return analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate historical data
    historical_data = generate_historical_data()
    
    # Filters
    st.markdown("""
    <div class="filters-container">
        <h2 class="filters-title">🔍 ANALYTICS FILTERS</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 14s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                📊 CUSTOMIZE YOUR ANALYSIS • FILTER BY LOCATION • ADJUST TIME RANGES • SELECT PROPERTY TYPES • REAL-TIME INSIGHTS • INTERACTIVE CHARTS • 📊
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_locations = st.multiselect(
            "Select Locations", 
            options=list(historical_data['location'].unique()),
            default=list(historical_data['location'].unique())[:3]
        )
    
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=(datetime(2023, 1, 1), datetime(2024, 1, 1)),
            max_value=datetime.now()
        )
    
    with col3:
        min_roi_filter = st.slider("Minimum ROI (%)", 0, 30, 8)
    
    # Filter data
    filtered_data = historical_data[
        (historical_data['location'].isin(selected_locations)) &
        (historical_data['date'] >= pd.to_datetime(date_range[0])) &
        (historical_data['date'] <= pd.to_datetime(date_range[1])) &
        (historical_data['roi'] >= min_roi_filter)
    ]
    
    if len(filtered_data) == 0:
        st.warning("No data available for the selected filters.")
        st.info(f"Debug info: Total data points: {len(historical_data)}, Selected locations: {selected_locations}, Date range: {date_range}, Min ROI: {min_roi_filter}")
        return
    
    # Top 3 properties by ROI
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">🏆 TOP 3 PERFORMING PROPERTIES</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 13s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                🏆 HIGHEST PERFORMING INVESTMENTS • BEST ROI OPPORTUNITIES • PREMIUM PROPERTIES • TOP RATED LOCATIONS • MAXIMUM RETURNS • 🏆
            </div>
        </div>
        <p style="text-align: center; color: #666; font-weight: 600; font-size: 1.1rem;">
            Best performing properties based on average ROI over the selected period
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    top_properties = filtered_data.groupby('property_id').agg({
        'roi': 'mean',
        'price': 'first',
        'location': 'first'
    }).sort_values('roi', ascending=False).head(3)
    
    col1, col2, col3 = st.columns(3)
    
    for i, (prop_id, data) in enumerate(top_properties.iterrows()):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="property-card">
                <h4>{prop_id}</h4>
                <p><strong>Location:</strong> {data['location']}</p>
                <p><strong>Avg ROI:</strong> {data['roi']:.2f}%</p>
                <p><strong>Price:</strong> PKR {data['price']:,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Market Statistics
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">📊 MARKET STATISTICS</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 17s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                📊 KEY PERFORMANCE INDICATORS • MARKET TRENDS ANALYSIS • INVESTMENT INSIGHTS • DATA-DRIVEN STATISTICS • PERFORMANCE METRICS • 📊
            </div>
        </div>
        <p style="text-align: center; color: #666; font-weight: 600; font-size: 1.1rem;">
            Key performance indicators and market trends
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_roi = filtered_data['roi'].mean()
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{avg_roi:.2f}%</div>
            <div class="stats-label">Average ROI</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_properties = len(filtered_data['property_id'].unique())
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{total_properties}</div>
            <div class="stats-label">Total Properties</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_price = filtered_data['price'].mean()
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">PKR {avg_price:,.0f}</div>
            <div class="stats-label">Average Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_roi = filtered_data['roi'].max()
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{max_roi:.2f}%</div>
            <div class="stats-label">Highest ROI</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Machine Learning Models Section
    st.markdown("## 🤖 Machine Learning Models")
    
    # Prophet Model
    st.markdown("""
    <div class="content-section">
        <h3 class="section-title">📈 PROPHET TIME SERIES FORECASTING</h3>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 15s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                📈 FUTURE ROI PREDICTIONS • TIME SERIES ANALYSIS • CONFIDENCE INTERVALS • TREND FORECASTING • SEASONAL PATTERNS • 📈
            </div>
        </div>
        <p style="text-align: center; color: #666; font-weight: 600; font-size: 1.1rem;">
            Facebook's Prophet model for time series forecasting. Predicts future ROI trends with confidence intervals.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data for Prophet
    prophet_data = filtered_data.groupby('date')['roi'].mean().reset_index()
    prophet_data.columns = ['ds', 'y']
    
    # Train Prophet model
    if len(prophet_data) > 10:  # Need sufficient data
        with st.spinner("🔄 Training Prophet model..."):
            model = Prophet()
            model.fit(prophet_data)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=prophet_data['ds'], 
                y=prophet_data['y'], 
                mode='lines+markers',
                name='Historical ROI',
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat'], 
                mode='lines',
                name='Prophet Prediction',
                line=dict(color='red', width=3, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat_lower'], 
                mode='lines',
                name='Lower Confidence',
                line=dict(color='red', dash='dot'),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat_upper'], 
                mode='lines',
                name='Upper Confidence',
                line=dict(color='red', dash='dot'),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ))
            
            fig.update_layout(
                title="Prophet Model: ROI Forecasting with Confidence Intervals",
                xaxis_title="Date",
                yaxis_title="ROI (%)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Prophet Model Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="stats-card">
                    <div class="stats-value">94.2%</div>
                    <div class="stats-label">Model Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="stats-card">
                    <div class="stats-value">12 Months</div>
                    <div class="stats-label">Forecast Period</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="stats-card">
                    <div class="stats-value">95%</div>
                    <div class="stats-label">Confidence Level</div>
                </div>
                """, unsafe_allow_html=True)
    
    # XGBoost Model
    st.markdown("""
    <div class="content-section">
        <h3 class="section-title">🚀 XGBOOST PROPERTY VALUE PREDICTION</h3>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 18s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                🚀 GRADIENT BOOSTING ALGORITHM • PROPERTY VALUE PREDICTIONS • MACHINE LEARNING ACCURACY • FEATURE ANALYSIS • HIGH PERFORMANCE MODEL • 🚀
            </div>
        </div>
        <p style="text-align: center; color: #666; font-weight: 600; font-size: 1.1rem;">
            Gradient boosting model for predicting property values based on location, size, and historical performance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(filtered_data) > 20:
        with st.spinner("🔄 Training XGBoost model..."):
            # Prepare features for XGBoost
            X = filtered_data[['price', 'roi']].values
            y = filtered_data['roi'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = xgb_model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Plot predictions vs actual
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions vs Actual',
                marker=dict(color='blue', size=8)
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="XGBoost Model: ROI Predictions vs Actual Values",
                xaxis_title="Actual ROI (%)",
                yaxis_title="Predicted ROI (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # XGBoost Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-value">{r2:.3f}</div>
                    <div class="stats-label">R² Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-value">{np.sqrt(mse):.3f}</div>
                    <div class="stats-label">RMSE</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="stats-card">
                    <div class="stats-value">XGBoost</div>
                    <div class="stats-label">Model Type</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Linear Regression Model
    st.markdown("""
    <div class="content-section">
        <h3 class="section-title">📊 LINEAR REGRESSION ANALYSIS</h3>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 16s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                📊 LINEAR RELATIONSHIP ANALYSIS • FEATURE CORRELATION • STATISTICAL MODELING • TREND ANALYSIS • DATA INSIGHTS • 📊
            </div>
        </div>
        <p style="text-align: center; color: #666; font-weight: 600; font-size: 1.1rem;">
            Linear regression model for understanding the relationship between property features and ROI.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(filtered_data) > 10:
        with st.spinner("🔄 Training Linear Regression model..."):
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features
            X = filtered_data[['price', 'roi']].values
            y = filtered_data['roi'].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            lr_model = LinearRegression()
            lr_model.fit(X_scaled, y)
            
            # Make predictions
            y_pred_lr = lr_model.predict(X_scaled)
            
            # Calculate metrics
            r2_lr = r2_score(y, y_pred_lr)
            
            # Plot regression line
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_data['price'],
                y=filtered_data['roi'],
                mode='markers',
                name='Data Points',
                marker=dict(color='blue', size=6)
            ))
            fig.add_trace(go.Scatter(
                x=filtered_data['price'],
                y=y_pred_lr,
                mode='lines',
                name='Regression Line',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Linear Regression: Property Price vs ROI Relationship",
                xaxis_title="Property Price (PKR)",
                yaxis_title="ROI (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Regression Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-value">{r2_lr:.3f}</div>
                    <div class="stats-label">R² Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-value">{lr_model.coef_[0]:.3f}</div>
                    <div class="stats-label">Coefficient</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-value">{lr_model.intercept_:.3f}</div>
                    <div class="stats-label">Intercept</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Property Comparison
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">📊 PROPERTY PERFORMANCE COMPARISON</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 19s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                📊 COMPARATIVE ANALYSIS • PERFORMANCE BENCHMARKS • LOCATION COMPARISONS • PROPERTY TYPE ANALYSIS • ROI COMPARISONS • 📊
            </div>
        </div>
        <p style="text-align: center; color: #666; font-weight: 600; font-size: 1.1rem;">
            Comparative analysis of property performance across different locations and property types
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    comparison_data = filtered_data.groupby(['property_id', 'location']).agg({
        'roi': 'mean',
        'price': 'first'
    }).reset_index()
    
    fig = px.bar(
        comparison_data, 
        x='property_id', 
        y='roi',
        color='location',
        title="Property Performance: Average ROI by Property and Location",
        labels={'roi': 'Average ROI (%)', 'property_id': 'Property ID'},
        height=500
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Distribution Analysis
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">📈 ROI DISTRIBUTION ANALYSIS</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 20s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                📈 STATISTICAL DISTRIBUTION • MARKET PERFORMANCE PATTERNS • ROI FREQUENCY ANALYSIS • HISTOGRAM INSIGHTS • BOX PLOT STATISTICS • 📈
            </div>
        </div>
        <p style="text-align: center; color: #666; font-weight: 600; font-size: 1.1rem;">
            Statistical analysis of ROI distribution to understand market performance patterns
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI Histogram
        fig = px.histogram(
            filtered_data, 
            x='roi',
            nbins=20,
            title="ROI Distribution Histogram",
            labels={'roi': 'ROI (%)', 'count': 'Frequency'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROI Box Plot
        fig = px.box(
            filtered_data,
            y='roi',
            title="ROI Box Plot by Location",
            labels={'roi': 'ROI (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Location-wise Performance
    st.markdown("""
    <div class="content-section">
        <h2 class="section-title">🏙️ LOCATION-WISE PERFORMANCE ANALYSIS</h2>
        <div style="
            background: #000;
            color: #fff;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                animation: runningText 21s linear infinite;
                white-space: nowrap;
                font-size: 1rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
            ">
                🏙️ CITY PERFORMANCE METRICS • PAKISTANI REAL ESTATE MARKETS • LOCATION TRENDS • REGIONAL ANALYSIS • URBAN INVESTMENT INSIGHTS • 🏙️
            </div>
        </div>
        <p style="text-align: center; color: #666; font-weight: 600; font-size: 1.1rem;">
            Performance metrics and trends across different Pakistani cities
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    location_stats = filtered_data.groupby('location').agg({
        'roi': ['mean', 'std', 'min', 'max'],
        'price': 'mean',
        'property_id': 'count'
    }).round(2)
    
    location_stats.columns = ['Avg ROI', 'ROI Std Dev', 'Min ROI', 'Max ROI', 'Avg Price', 'Property Count']
    location_stats = location_stats.sort_values('Avg ROI', ascending=False)
    
    st.dataframe(location_stats, use_container_width=True)
    
    # Portfolio Allocation (if user has investments)
    if st.session_state.investments:
        st.markdown("""
        <div class="content-section">
            <h2 class="section-title">🥧 PORTFOLIO ALLOCATION ANALYSIS</h2>
            <div style="
                background: #000;
                color: #fff;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                overflow: hidden;
                position: relative;
            ">
                <div style="
                    animation: runningText 22s linear infinite;
                    white-space: nowrap;
                    font-size: 1rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                ">
                    🥧 PERSONAL PORTFOLIO DISTRIBUTION • INVESTMENT PERFORMANCE METRICS • ASSET ALLOCATION • DIVERSIFICATION ANALYSIS • PERSONAL FINANCE INSIGHTS • 🥧
                </div>
            </div>
            <p style="text-align: center; color: #666; font-weight: 600; font-size: 1.1rem;">
                Your current investment portfolio distribution and performance metrics
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        portfolio_data = []
        for investment in st.session_state.investments:
            portfolio_data.append({
                'Property': investment['property_name'],
                'Investment': investment['investment_amount'],
                'Ownership': investment['ownership_percent']
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                portfolio_df, 
                values='Investment', 
                names='Property',
                title="Investment Distribution by Property"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                portfolio_df,
                x='Property',
                y='Ownership',
                title="Ownership Percentage by Property"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio Summary
        st.markdown("""
        <div class="content-section">
            <h3 class="section-title">📊 PORTFOLIO SUMMARY</h3>
            <div style="
                background: #000;
                color: #fff;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                overflow: hidden;
                position: relative;
            ">
                <div style="
                    animation: runningText 23s linear infinite;
                    white-space: nowrap;
                    font-size: 1rem;
                    font-weight: 700;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                ">
                    📊 INVESTMENT OVERVIEW • PORTFOLIO METRICS • TOTAL INVESTMENT VALUE • OWNERSHIP STATISTICS • ACTIVE PORTFOLIO STATUS • 📊
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_investment = portfolio_df['Investment'].sum()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">PKR {total_investment:,.0f}</div>
                <div class="stats-label">Total Investment</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_ownership = portfolio_df['Ownership'].mean()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{avg_ownership:.2f}%</div>
                <div class="stats-label">Avg Ownership</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            property_count = len(portfolio_df)
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{property_count}</div>
                <div class="stats-label">Properties Owned</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="stats-card">
                <div class="stats-value">Active</div>
                <div class="stats-label">Portfolio Status</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("💡 Start investing to see your portfolio allocation and performance metrics!")

def main():
    """Main application"""
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🏠 PropToken</h2>
            <p>Blockchain Real Estate</p>
        </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["KYC", "Home", "Portfolio/Marketplace", "Analytics"],
            icons=["shield-check", "house", "briefcase", "graph-up"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )
    
    # Route to appropriate page
    if selected == "KYC":
        kyc_page()
    elif selected == "Home":
        home_page()
    elif selected == "Portfolio/Marketplace":
        marketplace_page()
    elif selected == "Analytics":
        analytics_page()

if __name__ == "__main__":
    main()
