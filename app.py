from markupsafe import Markup
import streamlit as st
import tensorflow as tf
import base64
from PIL import Image
import pymysql
from io import BytesIO
import pandas as pd
import plotly.express as px
import os



with open("icon/bg.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

with open("icon/image.jpg", "rb") as image_file1:
    encoded_string1 = base64.b64encode(image_file1.read()).decode()        

st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}    

        [data-testid="stSidebar"] {{
            background-image: url("data:image/jpg;base64,{encoded_string1}");
            background-size: cover;    
           
        }}
    </style>
    """,
    unsafe_allow_html=True
)


def grading():

    st.set_option('deprecation.showfileUploaderEncoding', False)
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

    # Connect to the database
    conn = pymysql.connect(host='localhost', user='root', password='P@55w0rd', db='tb_grades')
    cursor = conn.cursor()


    # Add sidebar with two options
    sidebar_options = ["Tobacco Classification", "Dashboard"]
    selected_sidebar_option = st.sidebar.selectbox("Select an option", sidebar_options)
    
    st.image("icon/images12.jpg", width=100)
    st.title("Tobacco Leaf Grading System")
    
    if selected_sidebar_option == "Tobacco Classification":

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
                    <div style="width: 85%; height: 0; background-color: #F7F6D2; position: relative;
                    padding-bottom: 60.50%; margin-top: 20px; margin-left: 45px">

                    </div>
                    """
                st.markdown(html_content, unsafe_allow_html=True) 
                st.write("")
                st.write("") 
                
            else:
                
                image_bytes = file.read()  
                image = Image.open(file)

                # Save the image to a directory
                image_path = os.path.join("images", file.name)
                image.save(image_path)

                predictions= import_and_predict(image, model)
                class_names=['A2L', 'B3L2', 'C2L', 'Cannot be graded', 'X4OY']
                image_base64 = base64.b64encode(image_bytes).decode()
                

                # Create the HTML code for the image
                image_html = f'<img src="data:image/jpg;base64,{image_base64}">'
                css = """
                    <style>
                        img {
                            height: 430px;
                            width: 100%;
                            object-fit: contain;
                        
                    </style>
                """
                html_content = f"""
                    <div style="width: 85%; height: 200px; background-color: #F7F6D2;
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
                .stButton > button {color: black; border: ; padding: 15px 32px; text-align: center; background-color: gold; width: 200px; height: 70px;
                    margin-left: 50px; font-size: 18px;
                    
                }
            </style>
            """
            st.markdown(button_style, unsafe_allow_html=True) 
        
            if col1.button("Extract Features"): 
                
                try:
                    
                    # Create the HTML code for the image
                    image_html = f'<img src="data:image/jpg;base64,{image_base64}">'

                    class_names=['A2L', 'B3L2', 'C2L', 'Cannot be graded', 'X4OY']
                    class_index = 0
                    class_name = class_names[class_index]
                    g_names = class_names[np.argmax(predictions)]  
                    
                    if g_names == 'A2L':
                        msg = "Gold lemon (BL) coloured cured tobacco leaf, high quality leaf"
                        st.write("")
                        st.success(msg)
                        st.markdown("""       
                            ###### Properties the leaf
                            Flavor: Lemon (BL) colored tobacco leaves are generally considered to have a mild, sweet flavor profile with notes of citrus.
                            Aroma: The aroma of lemon (BL) tobacco leaves is often described as fresh and bright, with hints of citrus and sweetness.
                            Burn: Lemon (BL) tobacco leaves are known for their slow and even burn, producing a consistent and smooth smoke.

                            Appearance: Lemon (BL) tobacco leaves have a yellowish-green color, with a smooth and silky texture. They are generally considered to be visually appealing and are often used in high-quality cigars.

                            """)
                    elif g_names == 'B3L2':
                        msg = "Red Brown (RB) coloured cured tobacco leaf, one of the best quality tobacco"
                        st.write("")
                        st.success(msg)
                        st.markdown("""
                            ###### Properties the leaf     
                            Flavor: Red Brown (RB) colored tobacco leaves are often described as having a rich, sweet, and slightly spicy flavor profile.
                            Aroma: The aroma of Red Brown (RB) tobacco leaves is usually described as warm and earthy, with hints of sweetness and spices.
                            Burn: Red Brown (RB) tobacco leaves are known for their slow and consistent burn, producing a smooth and even smoke.

                            Appearance: Red Brown (RB) tobacco leaves have a deep red-brown color, with a smooth and silky texture. They are often used in premium cigars and pipe blends, contributing to the overall depth and complexity of the flavor profile.

                            """)
                    elif g_names == 'C2L':
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
                        msg = ""
                        st.write("This is not a virginia cured tobacco leaf")
                        st.error(msg)
                       

                    elif g_names == 'X4OY':
                        msg = "This is not a virginia cured tobacco leaf"
                        st.write("")
                        st.error(msg) 
                       
                    else :
                        msg = "This is not a cured tobacco leaf"
                        st.write("")
                        st.error(msg)
                                   
                except NameError:
                    
                    string = "Please Upload a cured tobacco leaf"
                    st.error(string)
                    
            if col2.button("Grade Tobacco"):
                
                
                try:
                    
                    # Create the HTML code for the image
                    image_html = f'<img src="data:image/jpg;base64,{image_base64}">'

                    class_names=['A2L', 'B3L2', 'C2L', 'Cannot be graded', 'X4OY']
                    class_index = 0
                    class_name = class_names[class_index]
                                        
                    st.write("")
                    g_names = class_names[np.argmax(predictions)]
                    if (g_names == 'X4OY' or g_names == 'Cannot be graded'):

                        string = "Cannot be graded !"
                        string1 = "Try to upload an image containing a cured tobacco leaf"
                        st.error(string)
                        st.info(string1)
                    else :    

                        string="Grading for cured tobacco leaf is grade :  "+class_names[np.argmax(predictions)]
                        st.success(string)

                    # Insert the image path and grade into the database
                    insert_query = "INSERT INTO images (image, grade) VALUES (%s, %s)"
                    cursor.execute(insert_query, (image_path, g_names))
                    conn.commit()
                    
                except NameError:
                    string = "Please Upload a cured tobacco leaf"
                    st.error(string)
                    
        else:
            st.info("Grading using a Camera")
            file = st.camera_input("Take a picture")
            
            if file is not None:
                
                # To read image file buffer with OpenCV:
                image_bytes = file.read()
                image = Image.open(file)

                # Save the image to a directory
                image_path = os.path.join("images", file.name)
                image.save(image_path)

                predictions= import_and_predict(image, model)
                class_names=['A2L', 'B3L2', 'C2L', 'Cannot be graded', 'X4OY']
                image_base64 = base64.b64encode(image_bytes).decode()
            
            else:
                st.write("")
                st.error("Press the take picture button")
            
            col1, col2 = st.columns(2)
                
            button_style = """
                <style>
                .stButton > button {color: black; border: ; padding: 15px 32px; text-align: center; background-color: gold; width: 200px; height: 70px;
                    margin-left: 50px; font-size: 18px;
                    
                }
            </style>
            """
            st.markdown(button_style, unsafe_allow_html=True)     
            
            if col1.button("Extract Features"):

                try:
                    
                    class_names=['A2L', 'B3L2', 'C2L', 'Cannot be graded', 'X4OY']
                    class_index = 0
                    class_name = class_names[class_index]
                    g_names = class_names[np.argmax(predictions)]  
                    
                    if g_names == 'A2L':
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
                    elif g_names == 'B3L2':
                        msg = "Red Brown (RB) coloured cured tobacco leaf, one of the best quality tobacco"
                        st.write("")
                        st.success(msg)
                        st.markdown("""
                            ###### Properties the leaf     
                            Flavor: Red Brown (RB) colored tobacco leaves are often described as having a rich, sweet, and slightly spicy flavor profile.
                            Aroma: The aroma of Red Brown (RB) tobacco leaves is usually described as warm and earthy, with hints of sweetness and spices.
                            Burn: Red Brown (RB) tobacco leaves are known for their slow and consistent burn, producing a smooth and even smoke.

                            Appearance: Red Brown (RB) tobacco leaves have a deep red-brown color, with a smooth and silky texture. They are often used in premium cigars and pipe blends, contributing to the overall depth and complexity of the flavor profile.

                            """)
                    elif g_names == 'C2L':
                        msg = "Orange (BF) coloured cured tobacco leaf, also best quality"
                        st.write("")
                        st.success(msg)
                        st.markdown("""
                            ###### Properties
                            Flavor: Orange (BF) colored tobacco leaves are often described as having a sweet and fruity flavor profile, with hints of citrus and a subtle spice.
                            Aroma: The aroma of Orange (BF) tobacco leaves is usually described as bright and fruity, with notes of orange and a touch of spice.
                            Burn: Orange (BF) tobacco leaves are known for their slow and consistent burn, producing a smooth and even smoke.

                            Appearance: Orange (BF) tobacco leaves have a bright orange color, with a smooth and silky texture. They are often used in premium cigars and pipe blends, contributing to the overall flavor and aroma of the product.

                            """)
                    elif g_names == 'Cannot be graded':
                        
                        st.error("This is not a virginia cured tobacco leaf")
                       
                    elif g_names == 'X4OY':
                        msg = "This is not a virginia cured tobacco leaf"
                        st.write("")
                        st.error(msg) 
                       
                    else :
                        msg = "This is not a cured tobacco leaf"
                        st.write("")
                        st.error(msg)   
                        
                except NameError:
                    st.write("")
                    
            if col2.button("Grade Tobacco"):
                
                try:


                    class_names=['A2L', 'B3L2', 'C2L', 'Cannot be graded', 'X4OY']
                    class_index = 0
                    st.write("")
                    g_names = class_names[np.argmax(predictions)]
                    if (g_names == 'X4OY' or g_names == 'Cannot be graded'):

                        string = "Cannot be graded !"
                        string1 = "Try to upload an image containing a cured tobacco leaf"
                        st.error(string)
                        st.info(string1)
                    else :    

                        string="Grading for cured tobacco leaf is grade :  "+class_names[np.argmax(predictions)]
                        st.success(string) 

                    # Insert the image path and grade into the database
                    insert_query = "INSERT INTO images (image, grade) VALUES (%s, %s)"
                    cursor.execute(insert_query, (image_path, g_names))
                    conn.commit()

                    # Close the cursor and connection
                    cursor.close()
                    conn.close()
                
                except NameError:
                    st.write("")

                
    elif selected_sidebar_option == "Dashboard":
        
        from st_aggrid import AgGrid
        from st_aggrid.grid_options_builder import GridOptionsBuilder
        import pandas as pd
        from IPython.display import display

        st.success("This is the dashboard content.")

        # Connect to the database
        #conn = pymysql.connect(host='localhost', user='root', password='P@55w0rd', db='tb_grades')
        cursor = conn.cursor()

        # Get data from the database
        #cursor.execute("SELECT id, image, grade FROM images")
        data = []
        for row in cursor:
            try:
                image_path = row[1]
                image = Image.open(image_path)
                data.append({'id': row[0], 'image': image, 'grade': row[2]})
            except:
                data.append({'id': row[0], 'image': 'Error', 'grade': row[2]})

        # Close the cursor and connection
        cursor.close()
        conn.close()

        # Display the data in a table
        pd.set_option('display.max_colwidth', 2000)
        df = pd.DataFrame(data)

        # Create a bar chart of the grade counts by id
        id_counts = df.groupby('grade')['id'].count()
        fig1 = px.bar(id_counts, x=id_counts.index, y=id_counts.values)

        # Create a pie chart of the grades
        grade_counts = df['grade'].value_counts()
        fig2 = px.pie(grade_counts, values=grade_counts.values, names=grade_counts.index)

        # Display the charts side by side
        col1, col2, = st.columns(2)
        with col1:
            st.plotly_chart(fig1)
        with col2:
            st.plotly_chart(fig2)

        st.write("Displaying data using Aggrid")
        gd = GridOptionsBuilder.from_dataframe(df)
        gd.configure_pagination(enabled=True)
        gd.configure_default_column(editable=True, groupable=True)

        gridoptions = gd.build()

        # Display the images in a table
        for i, row in df.iterrows():
            image = row['image']
            if not isinstance(image, str):
                display(image)

        AgGrid(df, gridOptions=gridoptions)


def main():
    
    grading()
    

if __name__ == "__main__":
    main()               