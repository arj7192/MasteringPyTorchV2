import UIKit

class PreviewViewController: UIViewController {
    @IBOutlet var imageView: UIImageView!
    @IBOutlet var resultView: UITextView!
    
    var capturedImage: UIImage?
    
    private lazy var module: TorchModule = {
        if let filePath = Bundle.main.path(forResource: "model", ofType: "pt"),
            let module = TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't find the model file!")
        }
    }()
    
    private lazy var labels: [String] = {
        if let filePath = Bundle.main.path(forResource: "digits", ofType: "txt"),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Can't find the text file!")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        imageView.image = capturedImage
        guard let resizedImage = capturedImage?.resized(to: CGSize(width: 28, height: 28)),
              var pixelBuffer = resizedImage.grayscaleNormalized() else {
            return
        }
//        imageView.image = resizedImage
        
        guard let outputs = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer)) else {
            return
        }

        print("Raw Predictions: \(outputs)") // Print the raw predictions array
        
        // Find the index of the maximum value in the outputs array
        if let maxIndex = outputs.indices.max(by: { outputs[$0].floatValue < outputs[$1].floatValue }) {
            let predictedDigit = maxIndex // This is the predicted digit
            print("Predicted Digit: \(predictedDigit)")
            resultView.text = "Predicted Digit: \(predictedDigit)"
        } else {
            print("Unable to determine predicted digit")
            resultView.text = "Unable to determine predicted digit"
        }
    }

}
