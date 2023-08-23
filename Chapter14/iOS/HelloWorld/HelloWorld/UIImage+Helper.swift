import UIKit

extension UIImage {
    func resized(to newSize: CGSize, scale: CGFloat = 1) -> UIImage {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = scale
        let renderer = UIGraphicsImageRenderer(size: newSize, format: format)
        let image = renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: newSize))
        }
        return image
    }

    func grayscaleNormalized() -> [Float32]? {
            guard let cgImage = self.cgImage else {
                return nil
            }
            let w = cgImage.width
            let h = cgImage.height
            let bytesPerPixel = 4
            let bytesPerRow = bytesPerPixel * w
            let bitsPerComponent = 8
            var rawBytes: [UInt8] = [UInt8](repeating: 0, count: w * h * 4)
            rawBytes.withUnsafeMutableBytes { ptr in
                if let cgImage = self.cgImage,
                    let context = CGContext(data: ptr.baseAddress,
                                            width: w,
                                            height: h,
                                            bitsPerComponent: bitsPerComponent,
                                            bytesPerRow: bytesPerRow,
                                            space: CGColorSpaceCreateDeviceRGB(),
                                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
                    let rect = CGRect(x: 0, y: 0, width: w, height: h)
                    context.draw(cgImage, in: rect)
                }
            }
            var normalizedBuffer: [Float32] = [Float32](repeating: 0, count: w * h)
            // convert RGB to grayscale and normalize
            for i in 0 ..< w * h {
                let red = Float32(rawBytes[i * 4 + 0])
                let green = Float32(rawBytes[i * 4 + 1])
                let blue = Float32(rawBytes[i * 4 + 2])
                let grayscaleValue = (0.2989 * red + 0.5870 * green + 0.1140 * blue) / 255.0
                normalizedBuffer[i] = (grayscaleValue - 0.1302) / 0.3069
            }
            return normalizedBuffer
        }
}
