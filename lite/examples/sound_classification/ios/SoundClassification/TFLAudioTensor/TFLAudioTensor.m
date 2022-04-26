// Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

#import "TFLAudioTensor.h"
#import "TFLAudioError.h"
#import "TFLUtils.h"

@implementation TFLAudioTensor
- (instancetype)initWithAudioFormat:(TFLAudioFormat *)format sampleCount:(NSUInteger)sampleCount {
  self = [self init];
  if (self) {
    _audioFormat = format;
    _ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:sampleCount * format.channelCount];
  }
  return self;
}

- (BOOL)loadBuffer:(TFLFloatBuffer *)floatBuffer
            offset:(NSInteger)offset
              size:(NSInteger)size
             error:(NSError **)error {
  return [_ringBuffer loadBuffer:floatBuffer offset:offset size:floatBuffer.size error:error];
}

- (BOOL)loadAudioRecord:(TFLAudioRecord *)audioRecord withError:(NSError **)error {
  if (![self.audioFormat isEqual:audioRecord.audioFormat]) {
    [TFLUtils createCustomError:error
                       withCode:TFLAudioErrorCodeInvalidArgumentError
                    description:@"Audio format of TFLAudioRecord does not match the audio format "
                                @"of Tensor Audio. Please ensure that the channelCount and "
                                @"sampleRate of both audio formats are equal."];
  }

  NSUInteger sizeToLoad = audioRecord.audioFormat.channelCount * audioRecord.bufferSize;
  TFLFloatBuffer *buffer = [audioRecord readAtOffset:0 withSize:sizeToLoad error:error];
  
  if (!buffer) {
    return NO;
  }

 return [self loadBuffer:buffer offset:0 size:sizeToLoad error:error];
}

//- (TFLFloatBuffer *)buffer {
//  TFLFloatBuffer *floatBuffer = [[TFLFloatBuffer alloc] initWithSize:_ringBuffer.buffer.size];
////  //  NSLog(@"%d",self.buffer.size);
////
////  // Return buffer in correct order.
////  // Buffer's beginning is marked by nextIndex.
////  // Copy the first chunk starting at position nextindex to the destination buffer's
////  // beginning.
////  NSInteger endChunkSize = _ringBuffer.buffer.size - nextIndex;
////  memcpy(floatBuffer.data, _buffer.data + nextIndex, sizeof(float) * endChunkSize);
////
////  // Copy the next chunk starting at position 0 until next index to the destination buffer
////  // locations after the chunk size that was previously copied.
////  memcpy(floatBuffer.data + endChunkSize, _buffer.data, sizeof(float) * nextIndex);
////
//  return [_ringBuffer floatBuffer];
//}
@end
