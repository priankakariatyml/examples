// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit

class ViewController: UIViewController {
  // MARK: - Variables
  @IBOutlet weak var tableView: UITableView!

  private var soundClassifier: SoundClassifier!
  private var bufferSize: Int = 0
  private var probabilities: [Float32] = []
  private var audioRecord: AudioRecord?
  private var audioTensor: AudioTensor!


  // MARK: - View controller lifecycle methods

  override func viewDidLoad() {
    super.viewDidLoad()


    tableView.dataSource = self
    tableView.backgroundColor = .white
    tableView.tableFooterView = UIView()

    soundClassifier = SoundClassifier(modelFileName: "sound_classification", delegate: self)
    
    // This initializer defaults to channel count 1
    let audioFormat = AudioFormat(sampleRate: UInt(soundClassifier.sampleRate))
    
    audioTensor = AudioTensor(audioFormat: audioFormat, sampleCount: UInt(soundClassifier.sampleRate))
    
//    startAudioRecognition()
//    let flBuffer = TFLFloatBuffer(size: 10000)
    
//    print("Heello")
    
    // Initialize audio record and gmlAudio with same buffer size and audio format as GMLAudio.
    // Once we move to the
    do {
      audioRecord = try AudioRecord(audioFormat: audioFormat, sampleCount: UInt(soundClassifier.sampleRate))
      DispatchQueue.global(qos: .background).async {
        self.startAudioRecognition()
      }

    }
    catch {
      print(error.localizedDescription)
    }
  }

  // MARK: - Private Methods

  /// Starts tapping AuudioRecord and recognizing on the output buffers
  private func startAudioRecognition() {
//    let flBuffer = TFLFloatBuffer(size: 10000)

    audioRecord?.startRecording { error in
      print(error?.localizedDescription)
      while true {
      do {
        
        try self.audioTensor.loadAudioRecord(audioRecord: self.audioRecord!)
        //      let audioTensorBuffer = TFLFloatBuffer(size: 40000)
        let audioTensorBuffer = self.audioTensor.ringBuffer.floatBuffer()
        print("DATA \(audioTensorBuffer.data[0])")

        self.soundClassifier.start(inputBuffer: audioTensorBuffer)
        
        //      TFLFloatBuffer *floatBuffer = [[TFLFloatBuffer alloc] initWithSize:_buffer.size];
        //      //  NSLog(@"%d",self.buffer.size);
        //
        //      // Return buffer in correct order.
        //      // Buffer's beginning is marked by nextIndex.
        //      // Copy the first chunk starting at position nextindex to the destination buffer's
        //      // beginning.
        //      NSInteger endChunkSize = _buffer.size - nextIndex;
        //      memcpy(floatBuffer.data, _buffer.data + nextIndex, sizeof(float) * endChunkSize);
        //
        //      // Copy the next chunk starting at position 0 until next index to the destination buffer
        //      // locations after the chunk size that was previously copied.
        //      memcpy(floatBuffer.data + endChunkSize, _buffer.data, sizeof(float) * nextIndex);
        
      }
      
      catch {
        print(error.localizedDescription)
      }
      }
  }

//  DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
//    for i in 0..<100000 {
//
//
////    print("Sum of times: \(time1 + time2)")
//  }
//  }
//
   
}
}

extension ViewController {
  func audioInputManagerDidFailToAchievePermission(_ audioInputManager: AudioInputManager) {
    let alertController = UIAlertController(
      title: "Microphone Permissions Denied",
      message: "Microphone permissions have been denied for this app. You can change this by going to Settings",
      preferredStyle: .alert
    )

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { _ in
      UIApplication.shared.open(
        URL(string: UIApplication.openSettingsURLString)!,
        options: [:],
        completionHandler: nil
      )
    }
    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)
  }
}

extension ViewController: SoundClassifierDelegate {
  func soundClassifier(
    _ soundClassifier: SoundClassifier,
    didInterpreteProbabilities probabilities: [Float32]
  ) {
    self.probabilities = probabilities
    DispatchQueue.main.async {
      print(self.probabilities)
      self.tableView.reloadData()
    }
  }
}

// MARK: - UITableViewDataSource
extension ViewController: UITableViewDataSource {
  func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
    return probabilities.count
  }

  func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    guard let cell = tableView.dequeueReusableCell(
      withIdentifier: "probabilityCell",
      for: indexPath
    ) as? ProbabilityTableViewCell else { return UITableViewCell() }

    cell.label.text = soundClassifier.labelNames[indexPath.row]
    UIView.animate(withDuration: 0.4) {
      cell.progressView.setProgress(self.probabilities[indexPath.row], animated: true)
    }
    return cell
  }
}
