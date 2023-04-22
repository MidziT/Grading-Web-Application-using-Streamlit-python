import streamlit as st
import tensorflow as tf
import base64
from PIL import Image

def grading():
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    #@st.cache(allow_output_mutation=True)
    @st.cache_resource
    def load_model(): 
        
        # Load the saved model
        model = tf.keras.models.load_model('model/my_model.h5', compile = False)
        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        # Compile the model with new settings
        return model

    model=load_model()
    st.title("Tobacco Leaf Grading System")
    
    options = ["Use File Uploader", "Use camera"]
    selected_option = st.selectbox("Select an option", options)
    
    from PIL import Image, ImageOps
    import numpy as np
    import io
    def import_and_predict(image_data, model):
            
        size = (180,180)
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        
        return prediction
            

    if selected_option == "Use File Uploader":
    
        file = st.file_uploader("Please upload a cured tobacco leaf image", type=["jpg", "png"])   
        
        
        if file is None:
            #st.text("Please upload an image file")
            html_content = """
                <div style="width: 85%; height: 0; background-color: grey; position: relative;
                padding-bottom: 60.50%; margin-top: 20px; margin-left: 45px">

                </div>
                """
            st.markdown(html_content, unsafe_allow_html=True) 
            st.write("")
            st.write("") 
            
        else:
            
            image_bytes = file.read()  
            image = Image.open(file)
            predictions= import_and_predict(image, model)
            class_names=['A', 'B', 'C', 'Cannot be graded', 'No Grade']
            image_base64 = base64.b64encode(image_bytes).decode()

            # Create the HTML code for the image
            image_html = f'<img src="data:image/jpg;base64,{image_base64}">'
            css = """
                <style>
                    img {
                        height: auto;
                        width: 100%;
                        object-fit: contain;
                    }
                </style>
            """
            html_content = f"""
                <div style="width: 85%; height: 200px; background-color: black;
                padding-bottom: 60.50%; margin-top: 20px; margin-left: 20px; margin-bottom:0">
                    {image_html}
                </div>
            """
            st.markdown(html_content + css, unsafe_allow_html=True)
            st.write("")
            st.write("")
            
        col1, col2 = st.columns(2)
        
        button_style = """
            <style>
            .stButton > button {color: black; border: ; padding: 15px 32px; text-align: center; background-color: white; width: 200px; height: 70px;
                margin-left: 50px; font-size: 18px;
                
            }
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True) 
    
        if col1.button("Extract Features"): 
            
            try:
                
                # Create the HTML code for the image
                image_html = f'<img src="data:image/jpg;base64,{image_base64}">'

                class_names=['A', 'B', 'C', 'Cannot be graded', 'No Grade']
                class_index = 0
                class_name = class_names[class_index]
                g_names = class_names[np.argmax(predictions)]  
                
                if g_names == 'A':
                    msg = "Gold lemon (BL) coloured cured tobacco leaf, high quality leaf"
                    st.write("")
                    st.write(msg)
                    st.markdown("""       
                        ###### Properties the leaf
                        Flavor: Lemon (BL) colored tobacco leaves are generally considered to have a mild, sweet flavor profile with notes of citrus.
                        Aroma: The aroma of lemon (BL) tobacco leaves is often described as fresh and bright, with hints of citrus and sweetness.
                        Burn: Lemon (BL) tobacco leaves are known for their slow and even burn, producing a consistent and smooth smoke.

                        Appearance: Lemon (BL) tobacco leaves have a yellowish-green color, with a smooth and silky texture. They are generally considered to be visually appealing and are often used in high-quality cigars.

                        """)
                elif g_names == 'B':
                    msg = "Red Brown (RB) coloured cured tobacco leaf, one of the best quality tobacco"
                    st.write("")
                    st.write(msg)
                    st.markdown("""
                        ###### Properties the leaf     
                        Flavor: Red Brown (RB) colored tobacco leaves are often described as having a rich, sweet, and slightly spicy flavor profile.
                        Aroma: The aroma of Red Brown (RB) tobacco leaves is usually described as warm and earthy, with hints of sweetness and spices.
                        Burn: Red Brown (RB) tobacco leaves are known for their slow and consistent burn, producing a smooth and even smoke.

                        Appearance: Red Brown (RB) tobacco leaves have a deep red-brown color, with a smooth and silky texture. They are often used in premium cigars and pipe blends, contributing to the overall depth and complexity of the flavor profile.

                        """)
                elif g_names == 'C':
                    msg = "Orange (BF) coloured cured tobacco leaf, also best quality"
                    st.write("")
                    st.write(msg)
                    st.markdown("""
                        ###### Properties
                        Flavor: Orange (BF) colored tobacco leaves are often described as having a sweet and fruity flavor profile, with hints of citrus and a subtle spice.
                        Aroma: The aroma of Orange (BF) tobacco leaves is usually described as bright and fruity, with notes of orange and a touch of spice.
                        Burn: Orange (BF) tobacco leaves are known for their slow and consistent burn, producing a smooth and even smoke.

                        Appearance: Orange (BF) tobacco leaves have a bright orange color, with a smooth and silky texture. They are often used in premium cigars and pipe blends, contributing to the overall flavor and aroma of the product.

                        """)
                elif g_names == 'Cannot be graded':
                    msg = "Over cured poor tobacco leaf"
                    st.write("")
                    st.write(msg)
                    st.markdown("""
                        ###### Properties
                        Flavor: Poor quality tobacco leaves often have a harsh, bitter, and unpleasantly pungent flavor profile.
                        Aroma: The aroma of poor quality tobacco leaves is often described as musty, damp, or stale, lacking the desirable characteristics of good quality tobacco.
                        Burn: Poor quality tobacco leaves are known for producing a harsh and uneven burn, with the possibility of burning too quickly or going out frequently.

                        Appearance: Poor quality tobacco leaves may be discolored, with a dull and rough texture, and may have visible signs of mold, mildew, or insect damage.
                        
                        """)
                else:
                    msg = "The leaf is ungraded or its not a cured tobacco leaf"
                    st.write("")
                    st.write(msg)
                    st.markdown("""
                        ###### Properties
                        Flavor: The flavor of ungraded tobacco leaves can be unpredictable and may range from bland to harsh, with little consistency from leaf to leaf.
                        Aroma: The aroma of ungraded tobacco leaves may also be inconsistent, and may range from unpleasant to neutral.
                        Burn: Ungraded tobacco leaves may burn poorly, with an uneven burn and the possibility of going out frequently.
                        Appearance: Ungraded tobacco leaves may have varying colors and textures or green leaves, with no consistent standards. They may also have visible signs of insect damage, mold, or mildew.
                        """)              
            except NameError:
                
                string = "Please Upload a cured tobacco leaf"
                st.warning(string)
                
            #st.success(string)
        if col2.button("Grade Tobacco"):
            
            
            try:
                
                # Create the HTML code for the image
                image_html = f'<img src="data:image/jpg;base64,{image_base64}">'

                class_names=['A', 'B', 'C', 'Cannot be graded', 'No Grade']
                class_index = 0
                class_name = class_names[class_index]
                                    
                st.write("")
                string="Grading for cured tobacco leaf is grade :  "+class_names[np.argmax(predictions)]
                g_names = class_names[np.argmax(predictions)]
                st.success(string)
                
            except NameError:
                string = "Please Upload a cured tobacco leaf"
                st.warning(string)
                
    else:
        st.info("Grading using a Camera")
        file = st.camera_input("Take a picture")
        
        if file is not None:
            
            # To read image file buffer with OpenCV:
            image_bytes = file.read()
            image = Image.open(file)
            predictions= import_and_predict(image, model)
            class_names=['A', 'B', 'C', 'Cannot be graded', 'No Grade']
            image_base64 = base64.b64encode(image_bytes).decode()
        
        else:
            st.write("")
            st.warning("Press the take picture button")
        
        col1, col2 = st.columns(2)
            
        button_style = """
            <style>
            .stButton > button {color: black; border: ; padding: 15px 32px; text-align: center; background-color: white; width: 200px; height: 70px;
                margin-left: 50px; font-size: 18px;
                
            }
        </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)     
        
        if col1.button("Extract Features"):

            try:
                
                class_names=['A', 'B', 'C', 'Cannot be graded', 'No Grade']
                class_index = 0
                class_name = class_names[class_index]
                g_names = class_names[np.argmax(predictions)]  
                
                if g_names == 'A':
                    msg = "Gold lemon (BL) coloured cured tobacco leaf, high quality leaf"
                    st.write("")
                    st.write(msg)
                    st.markdown("""       
                        ###### Properties the leaf
                        Flavor: Lemon (BL) colored tobacco leaves are generally considered to have a mild, sweet flavor profile with notes of citrus.
                        Aroma: The aroma of lemon (BL) tobacco leaves is often described as fresh and bright, with hints of citrus and sweetness.
                        Burn: Lemon (BL) tobacco leaves are known for their slow and even burn, producing a consistent and smooth smoke.

                        Appearance: Lemon (BL) tobacco leaves have a yellowish-green color, with a smooth and silky texture. They are generally considered to be visually appealing and are often used in high-quality cigars.

                        """)
                elif g_names == 'B':
                    msg = "Red Brown (RB) coloured cured tobacco leaf, one of the best quality tobacco"
                    st.write("")
                    st.write(msg)
                    st.markdown("""
                        ###### Properties the leaf     
                        Flavor: Red Brown (RB) colored tobacco leaves are often described as having a rich, sweet, and slightly spicy flavor profile.
                        Aroma: The aroma of Red Brown (RB) tobacco leaves is usually described as warm and earthy, with hints of sweetness and spices.
                        Burn: Red Brown (RB) tobacco leaves are known for their slow and consistent burn, producing a smooth and even smoke.

                        Appearance: Red Brown (RB) tobacco leaves have a deep red-brown color, with a smooth and silky texture. They are often used in premium cigars and pipe blends, contributing to the overall depth and complexity of the flavor profile.

                        """)
                elif g_names == 'C':
                    msg = "Orange (BF) coloured cured tobacco leaf, also best quality"
                    st.write("")
                    st.write(msg)
                    st.markdown("""
                        ###### Properties
                        Flavor: Orange (BF) colored tobacco leaves are often described as having a sweet and fruity flavor profile, with hints of citrus and a subtle spice.
                        Aroma: The aroma of Orange (BF) tobacco leaves is usually described as bright and fruity, with notes of orange and a touch of spice.
                        Burn: Orange (BF) tobacco leaves are known for their slow and consistent burn, producing a smooth and even smoke.

                        Appearance: Orange (BF) tobacco leaves have a bright orange color, with a smooth and silky texture. They are often used in premium cigars and pipe blends, contributing to the overall flavor and aroma of the product.

                        """)
                elif g_names == 'Cannot be graded':
                    msg = "Over cured poor tobacco leaf"
                    st.write("")
                    st.write(msg)
                    st.markdown("""
                        ###### Properties
                        Flavor: Poor quality tobacco leaves often have a harsh, bitter, and unpleasantly pungent flavor profile.
                        Aroma: The aroma of poor quality tobacco leaves is often described as musty, damp, or stale, lacking the desirable characteristics of good quality tobacco.
                        Burn: Poor quality tobacco leaves are known for producing a harsh and uneven burn, with the possibility of burning too quickly or going out frequently.

                        Appearance: Poor quality tobacco leaves may be discolored, with a dull and rough texture, and may have visible signs of mold, mildew, or insect damage.
                        
                        """)
                else:
                    msg = "The leaf is ungraded or its not a cured tobacco leaf"
                    st.write("")
                    st.write(msg)
                    st.markdown("""
                        ###### Properties
                        Flavor: The flavor of ungraded tobacco leaves can be unpredictable and may range from bland to harsh, with little consistency from leaf to leaf.
                        Aroma: The aroma of ungraded tobacco leaves may also be inconsistent, and may range from unpleasant to neutral.
                        Burn: Ungraded tobacco leaves may burn poorly, with an uneven burn and the possibility of going out frequently.
                        Appearance: Ungraded tobacco leaves may have varying colors and textures or green leaves, with no consistent standards. They may also have visible signs of insect damage, mold, or mildew.
                        """)
                    
                    
            except NameError:
                st.write("")
                
        if col2.button("Grade Tobacco"):
            
            try:

                class_names=['A', 'B', 'C', 'Cannot be graded', 'No Grade']
                class_index = 0
                class_name = class_names[class_index]
                st.write("")
                string="Grading for cured tobacco leaf is grade :  "+class_names[np.argmax(predictions)]
                g_names = class_names[np.argmax(predictions)]
                st.success(string)  
            
            except NameError:
                st.write("")
                
                        
def main():
    
    grading()

if __name__ == "__main__":
    main()               