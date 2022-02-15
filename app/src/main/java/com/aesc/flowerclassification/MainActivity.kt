package com.aesc.flowerclassification

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.min


class MainActivity : AppCompatActivity(), View.OnClickListener {
    private var tflite: Interpreter? = null
    private var inputImageBuffer: TensorImage? = null
    private var imageSizeX = 0
    private var imageSizeY = 0
    private var outputProbabilityBuffer: TensorBuffer? = null
    private var probabilityProcessor: TensorProcessor? = null
    private val IMAGE_MEAN = 0.0f
    private val IMAGE_STD = 1.0f
    private val PROBABILITY_MEAN = 0.0f
    private val PROBABILITY_STD = 255.0f
    private var bitmap: Bitmap? = null
    private var labels: List<String>? = null
    var imageuri: Uri? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageview.setOnClickListener(this)
        try {
            tflite = Interpreter(loadmodelfile(this))
        } catch (e: Exception) {
            e.printStackTrace()
        }

        classify.setOnClickListener(this)
    }

    private fun loadmodelfile(activity: Activity): ByteBuffer {
        val fileDescriptor = activity.assets.openFd("model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startoffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 12 && resultCode == RESULT_OK && data != null) {
            imageuri = data.data
            try {
                bitmap = MediaStore.Images.Media.getBitmap(contentResolver, imageuri)
                imageview.setImageBitmap(bitmap)
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
    }

    override fun onClick(v: View?) {
        when (v?.id) {
            R.id.imageview -> {
                val intent = Intent()
                intent.type = "image/*"
                intent.action = Intent.ACTION_GET_CONTENT
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), 12)
            }
            R.id.classify -> {
                val imageTensorIndex = 0
                val imageShape = tflite!!.getInputTensor(imageTensorIndex).shape() // {1, height, width, 3}

                imageSizeY = imageShape[1]
                imageSizeX = imageShape[2]
                val imageDataType: DataType = tflite!!.getInputTensor(imageTensorIndex).dataType()

                val probabilityTensorIndex = 0
                val probabilityShape =
                    tflite!!.getOutputTensor(probabilityTensorIndex).shape() // {1, NUM_CLASSES}

                val probabilityDataType: DataType =
                    tflite!!.getOutputTensor(probabilityTensorIndex).dataType()

                inputImageBuffer = TensorImage(imageDataType)
                outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)
                probabilityProcessor = TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build()

                inputImageBuffer = loadImage(bitmap!!)

                tflite!!.run(
                    inputImageBuffer?.buffer,
                    outputProbabilityBuffer?.buffer?.rewind()
                )
                showresult()
            }
            else -> Toast.makeText(this, "No se selecciono nada xd", Toast.LENGTH_LONG).show()
        }
    }

    private fun showresult() {
        try {
            labels = FileUtil.loadLabels(this, "dict.txt")
        } catch (e: java.lang.Exception) {
            e.printStackTrace()
        }
        val labeledProbability: Map<String, Float> =
            TensorLabel(labels!!, probabilityProcessor!!.process(outputProbabilityBuffer))
                .mapWithFloatValue
        val maxValueInMap: Float = Collections.max(labeledProbability.values)

        for ((key, value) in labeledProbability) {
            if (value == maxValueInMap) {
                classifytext.text = key
            }
        }
    }

    private fun loadImage(bitmap: Bitmap?): TensorImage {
        // Loads bitmap into a TensorImage.
        inputImageBuffer?.load(bitmap)

        // Creates processor for the TensorImage.
        val cropSize: Int = min(bitmap!!.width, bitmap.height)

        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        val imageProcessor: ImageProcessor = ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(getPreprocessNormalizeOp())
            .build()
        return imageProcessor.process(inputImageBuffer)
    }

    private fun getPreprocessNormalizeOp() = NormalizeOp(IMAGE_MEAN, IMAGE_STD)

    private fun getPostprocessNormalizeOp() = NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)
}
