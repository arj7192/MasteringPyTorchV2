import UIKit
import AVFoundation

class CaptureViewController: UIViewController, AVCapturePhotoCaptureDelegate {
    @IBOutlet var captureButton: UIButton!
    @IBOutlet var imageView: UIImageView!
    
    private var captureSession: AVCaptureSession!
    private var photoOutput: AVCapturePhotoOutput!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var capturedImage: UIImage?

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
    }
    
    func setupCamera() {
        captureSession = AVCaptureSession()
        guard let captureDevice = AVCaptureDevice.default(for: .video) else {
            fatalError("Cannot access camera.")
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: captureDevice)
            captureSession.addInput(input)
            
            photoOutput = AVCapturePhotoOutput()
            captureSession.addOutput(photoOutput)
            
            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            previewLayer.videoGravity = .resizeAspectFill // Maintain aspect ratio
            // Calculate the square frame that fits within the screen bounds
            let minSideLength = min(view.bounds.width, view.bounds.height)
            let previewFrame = CGRect(
                x: (view.bounds.width - minSideLength) / 2,
                y: (view.bounds.height - minSideLength) / 2,
                width: minSideLength,
                height: minSideLength
            )
            previewLayer.frame = previewFrame
            view.layer.addSublayer(previewLayer)
            
            captureSession.startRunning()
        } catch {
            fatalError("Cannot set up camera.")
        }
    }
    
    @IBAction func captureButtonTapped(_ sender: UIButton) {
        let settings = AVCapturePhotoSettings()
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
        
    // Inside photoOutput(_:didFinishProcessingPhoto:error:) function
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let imageData = photo.fileDataRepresentation(), let image = UIImage(data: imageData) {
            capturedImage = cropImage(image, to: previewLayer.frame)
            performSegue(withIdentifier: "showImagePreview", sender: self)
        }
    }

    func cropImage(_ image: UIImage, to frame: CGRect) -> UIImage? {
        let scale = UIScreen.main.scale
        let imageSize = image.size
        let previewSize = frame.size

        // Calculate scaling factors to map preview frame to image size
        let xScale = imageSize.width / previewSize.width
        let yScale = imageSize.height / previewSize.height
        
//        print("Image Size: \(imageSize)")
//        print("Preview Size: \(previewSize)")
//        print("Preview X: \(frame.origin.x)")
//        print("Preview Y: \(frame.origin.y)")
//        print("X Scale: \(xScale)")
//        print("Y Scale: \(yScale)")

        // Calculate the crop rect based on the visible portion of the preview layer
        let cropRect = CGRect(
            x: frame.origin.x,
            y: frame.origin.y,
            width: previewSize.width * xScale,
            height: previewSize.height * yScale
        )

        if let cgImage = image.cgImage?.cropping(to: cropRect) {
            return UIImage(cgImage: cgImage, scale: scale, orientation: image.imageOrientation)
        }

        return nil
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "showImagePreview", let previewVC = segue.destination as? PreviewViewController {
            previewVC.capturedImage = capturedImage
            capturedImage = nil // Reset the capturedImage property
        }
    }
}
