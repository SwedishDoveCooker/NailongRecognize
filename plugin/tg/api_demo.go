package main

import (
	"bytes"
	"fmt"
	"image"
	"image/gif"
	"net/http"
	"os"

	"github.com/disintegration/imaging"
	"github.com/gin-gonic/gin"
)

var (
	device string
	model  interface{}
)

func loadModel(modelPath string, device string) (interface{}, error) {
	fmt.Printf("Loading model from %s on device %s\n", modelPath, device)
	return nil, nil
}

func predictFrame(frame image.Image, model interface{}, transformFunc func(image.Image) image.Image) (bool, error) {
	_ = transformFunc
	return true, nil
}

func testTransform(img image.Image) image.Image {
	return imaging.Resize(img, 224, 224, imaging.Lanczos)
}

func predictSingleImage(file []byte) (string, error) {
	img, format, err := image.Decode(bytes.NewReader(file))
	if err != nil {
		return "", fmt.Errorf("failed to decode image: %v", err)
	}

	if format == "gif" {
		gifImg, err := gif.DecodeAll(bytes.NewReader(file))
		if err != nil {
			return "", fmt.Errorf("failed to decode GIF: %v", err)
		}
		for _, frame := range gifImg.Image {
			frameImg := imaging.New(frame.Bounds().Dx(), frame.Bounds().Dy(), frame.Palette)
			imaging.Paste(frameImg, imaging.Image(frame), image.Point{0, 0})
			if ok, _ := predictFrame(frameImg, model, testTransform); ok {
				return "Prediction: Positive", nil
			}
		}
		return "Prediction: Negative", nil
	}

	transformedImg := testTransform(img)
	result, err := predictFrame(transformedImg, model, testTransform)
	if err != nil {
		return "", err
	}
	if result {
		return "奶龙", nil
	}
	return "非奶龙", nil
}

func main() {
	modelPath := "./nailong.pth"
	device = "cuda"
	if _, ok := os.LookupEnv("NO_CUDA"); ok {
		device = "cpu"
	}
	var err error
	model, err = loadModel(modelPath, device)
	if err != nil {
		fmt.Printf("Failed to load model: %v\n", err)
		os.Exit(1)
	}

	r := gin.Default()

	r.POST("/predict", func(c *gin.Context) {
		file, err := c.FormFile("image")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "No image file provided"})
			return
		}

		fileData, err := file.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open image file"})
			return
		}
		defer fileData.Close()

		buf := new(bytes.Buffer)
		_, err = buf.ReadFrom(fileData)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read image file"})
			return
		}

		result, err := predictSingleImage(buf.Bytes())
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{"result": result})
	})

	r.Run(":7001")
}
