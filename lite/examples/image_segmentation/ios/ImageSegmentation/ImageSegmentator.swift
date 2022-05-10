// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlowLiteTaskVision
import UIKit

class ImageSegmentator {

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var imageSegmenter: ImageSegmenter

  /// Dedicated DispatchQueue for TF Lite operations.
  private let tfLiteQueue: DispatchQueue
  
  // MARK: - Initialization


  /// Create a new Image Segmentator instance.
  static func newInstance(completion: @escaping ((Result<ImageSegmentator>) -> Void)) {
    // Create a dispatch queue to ensure all operations on the Intepreter will run serially.
    let tfLiteQueue = DispatchQueue(label: "org.tensorflow.examples.lite.image_segmentation")

    // Run initialization in background thread to avoid UI freeze.
    tfLiteQueue.async {
      // Construct the path to the model file.
      guard
        let modelPath = Bundle.main.path(
          forResource: Constants.modelFileName,
          ofType: Constants.modelFileExtension
        )
      else {
        print(
          "Failed to load the model file with name: "
            + "\(Constants.modelFileName).\(Constants.modelFileExtension)")
        DispatchQueue.main.async {
          completion(
            .error(
              InitializationError.invalidModel(
                "\(Constants.modelFileName).\(Constants.modelFileExtension)"
              )))
        }
        return
      }

      let options = ImageSegmenterOptions(modelPath: modelPath)
      options.baseOptions.computeSettings.cpuSettings.numThreads = 2

      do {
        // Create the `Interpreter`.
        let imageSegmenter = try ImageSegmenter.imageSegmenter(options: options)
        
        let segmentator = ImageSegmentator(
          tfLiteQueue: tfLiteQueue,
          imageSegmenter: imageSegmenter
        )
        DispatchQueue.main.async {
          completion(.success(segmentator))
        }
      } catch let error {
        print("Failed to create the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(InitializationError.internalError(error)))
        }
        return
      }
    }
  }

  /// Initialize Image Segmentator instance.
  fileprivate init(
    tfLiteQueue: DispatchQueue,
    imageSegmenter: ImageSegmenter
  ) {
    // Store TF Lite intepreter
    self.imageSegmenter = imageSegmenter
    self.tfLiteQueue = tfLiteQueue
  }

  // MARK: - Image Segmentation

  /// Run segmentation on a given image.
  /// - Parameter image: the target image.
  /// - Parameter completion: the callback to receive segmentation result.
  func runSegmentation(
    _ image: UIImage, completion: @escaping ((Result<SegmentationResult>) -> Void)
  ) {
    tfLiteQueue.async {

      do {
        
        guard let mlImage = MLImage(image: image) else {
          DispatchQueue.main.async {
            completion(.error(SegmentationError.resultVisualizationError))
          }
          return
        }
        
        let segmentationResult = try self.imageSegmenter.segment(gmlImage: mlImage)
        
        print("First Colored Label:\(segmentationResult.segmentations[0].coloredLabels[0].label) Segmentation Mask Object: \(segmentationResult.segmentations[0].categoryMask)")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.resultVisualizationError))
        }
        
      } catch let error {
        print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
        DispatchQueue.main.async {
          completion(.error(SegmentationError.internalError(error)))
        }
        return
      }
    }
  }
}

// MARK: - Types

/// Callback type for image segmentation request.
typealias ImageSegmentationCompletion = (SegmentationResult?, Error?) -> Void

/// Representation of the image segmentation result.
struct SegmentationResult {
  /// Segmentation result as an array. Each value represents the most likely class the pixel
  /// belongs to.
  let array: [[Int]]

  /// Visualization of the segmentation result.
  let resultImage: UIImage

  /// Overlay the segmentation result on input image.
  let overlayImage: UIImage

  /// Processing time.
  let preprocessingTime: TimeInterval
  let inferenceTime: TimeInterval
  let postProcessingTime: TimeInterval
  let visualizationTime: TimeInterval

  /// Dictionary of classes found in the image, and the color used to represent the class in
  /// segmentation result visualization.
  let colorLegend: [String: UIColor]
}

/// Convenient enum to return result with a callback
enum Result<T> {
  case success(T)
  case error(Error)
}

/// Define errors that could happen in the initialization of this class
enum InitializationError: Error {
  // Invalid TF Lite model
  case invalidModel(String)

  // Invalid label list
  case invalidLabelList(String)

  // TF Lite Internal Error when initializing
  case internalError(Error)
}

/// Define errors that could happen in when doing image segmentation
enum SegmentationError: Error {
  // Invalid input image
  case invalidImage

  // TF Lite Internal Error when initializing
  case internalError(Error)

  // Invalid input image
  case resultVisualizationError
}

// MARK: - Constants
private enum Constants {
  /// Label list that the segmentation model detects.
  static let modelFileName = "deeplabv3"

  /// The TF Lite segmentation model file
  static let modelFileExtension = "tflite"

  /// List of colors to visualize segmentation result.
  static let legendColorList: [UInt32] = [
    0xFFFF_B300, // Vivid Yellow
    0xFF80_3E75, // Strong Purple
    0xFFFF_6800, // Vivid Orange
    0xFFA6_BDD7, // Very Light Blue
    0xFFC1_0020, // Vivid Red
    0xFFCE_A262, // Grayish Yellow
    0xFF81_7066, // Medium Gray
    0xFF00_7D34, // Vivid Green
    0xFFF6_768E, // Strong Purplish Pink
    0xFF00_538A, // Strong Blue
    0xFFFF_7A5C, // Strong Yellowish Pink
    0xFF53_377A, // Strong Violet
    0xFFFF_8E00, // Vivid Orange Yellow
    0xFFB3_2851, // Strong Purplish Red
    0xFFF4_C800, // Vivid Greenish Yellow
    0xFF7F_180D, // Strong Reddish Brown
    0xFF93_AA00, // Vivid Yellowish Green
    0xFF59_3315, // Deep Yellowish Brown
    0xFFF1_3A13, // Vivid Reddish Orange
    0xFF23_2C16, // Dark Olive Green
    0xFF00_A1C2, // Vivid Blue
  ]
}

