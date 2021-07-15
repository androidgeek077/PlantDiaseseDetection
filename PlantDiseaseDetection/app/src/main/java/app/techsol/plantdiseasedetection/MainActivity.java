package app.techsol.plantdiseasedetection;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;


public class MainActivity extends AppCompatActivity {
    String str;

    private Interpreter tflite;
    private TensorImage tensorImage;
    private Bitmap bitmap;
    private Uri selectedProfileImageUri;
    ImageView selectedImageIV;
    Button PredictBtn, sleectImageBtn;
    private static final int RC_PHOTO_PICKER = 1;
    TextView predictedDiseaseTV;

    ImageView profileImageView;
    private static List<Float> input_signal;
    private List<String> classList;
    private float max;
    String[] classNames = {"Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy", "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"};
    float[] resultProb;

    List<String> associatedAxisLabels = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        selectedImageIV = findViewById(R.id.selectedImageIV);
        input_signal = new ArrayList<Float>();

        final String ASSOCIATED_AXIS_LABELS = "labels.txt";

        try {
            associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading label file", e);
        }

        predictedDiseaseTV = findViewById(R.id.predictedDiseaseTV);
        sleectImageBtn = findViewById(R.id.sleectImageBtn);
        sleectImageBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getProfilePicture();
            }
        });
        PredictBtn = findViewById(R.id.PredictBtn);
        PredictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ImageProcessor imageProcessor =
                        new ImageProcessor.Builder()
                                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                                .build();

                tensorImage = new TensorImage(DataType.FLOAT32);
                tensorImage.load(bitmap);
                tensorImage = imageProcessor.process(tensorImage);
                TensorBuffer probabilityBuffer =
                        TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);

                try {
                    MappedByteBuffer tfliteModel
                            = FileUtil.loadMappedFile(MainActivity.this,
                            "model.tflite");
                    tflite = new Interpreter(tfliteModel);
                } catch (IOException e) {
                    Log.e("tfliteSupport", "Error reading model", e);
                }


// Running inference
                if (null != tflite) {
                    tflite.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer());
//                    float[] results = activityInference.getActivityProb(toFloatArray(input_signal));
                    float[] result = probabilityBuffer.getFloatArray();
                    getIndexOfLargest(result);

//                    TensorProcessor probabilityProcessor =
//                            new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();
//
//                    if (null != associatedAxisLabels) {
//                        // Map of labels and their corresponding probability
//                        TensorLabel labels = new TensorLabel(associatedAxisLabels,
//                                probabilityProcessor.process(probabilityBuffer));
//
//                        // Create a map to access the result based on label
//                        Map<String, Float> floatMap = labels.getMapWithFloatValue();
//                    }


                    resultProb = new float[result.length];

//                    String str = getApplication().getAssets().open("");
                    for (int i = 0; i < result.length; i++) {
                        resultProb[i] = result[i] / 255;
                    }
                    Toast.makeText(MainActivity.this, "" + getIndexOfLargest(result), Toast.LENGTH_SHORT).show();

                    if (classNames.length > getIndexOfLargest(result)) {
                        predictedDiseaseTV.setText("" + classNames[getIndexOfLargest(resultProb)]);
                        Toast.makeText(MainActivity.this, "" + classNames[getIndexOfLargest(result)], Toast.LENGTH_SHORT).show();
                    } else {
                        Toast.makeText(MainActivity.this, "Model is unable to predict the right disease", Toast.LENGTH_SHORT).show();
                    }
//                    Toast.makeText(MainActivity.this, ""+getMaxNo(result), Toast.LENGTH_SHORT).show();
//                    Toast.makeText(MainActivity.this, ""+getIndexOfLargest(result), Toast.LENGTH_SHORT).show();
//                    predictedDiseaseTV.setText(classList.get(getIndexOfLargest(result)) +"");
//                    final String ASSOCIATED_AXIS_LABELS = "labels.txt";
//                    List<String> associatedAxisLabels = null;
//
//                    try {
//                        associatedAxisLabels = FileUtil.loadLabels(MainActivity.this, ASSOCIATED_AXIS_LABELS);
//                    } catch (IOException e) {
//                        Log.e("tfliteSupport", "Error reading label file", e);
//                    }
//
//                    TensorProcessor probabilityProcessor =
//                            new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();
//
//                    if (null != associatedAxisLabels) {
//                        // Map of labels and their corresponding probability
//                        TensorLabel labels = new TensorLabel(associatedAxisLabels,
//                                probabilityProcessor.process(probabilityBuffer));
//
//                        // Create a map to access the result based on label
//                        Map<String, Float> floatMap = labels.getMapWithFloatValue();
//                        predictedDiseaseTV.setText(floatMap.toString());
//                    }
                }
            }
        });
    }

    public void getProfilePicture() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        intent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
        startActivityForResult(Intent.createChooser(intent, "Complete action using"), RC_PHOTO_PICKER);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            Uri selectedImageUri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
            } catch (IOException e) {
                e.printStackTrace();
            }

            selectedProfileImageUri = selectedImageUri;
            selectedImageIV.setImageURI(selectedImageUri);
            selectedImageIV.setVisibility(View.VISIBLE);
        }

    }

    public int getIndexOfLargest(float[] array) {
        if (array == null || array.length == 0) return -1; // null or empty

        int largest = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[largest]) largest = i;
        }
        return largest; // position of the first largest found
    }

    float getMaxNo(float decMax[]) {
        max = 0.0f;
        for (int counter = 1; counter < decMax.length; counter++) {
            if (decMax[counter] > max) {
                max = decMax[counter];
            }
        }
        return max;
    }
}