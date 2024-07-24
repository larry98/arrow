// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>

#include "arrow/compute/kernels/codegen_internal.h"
#include "arrow/visit_data_inline.h"

namespace arrow {

using internal::checked_cast;

namespace compute {
namespace internal {

struct KeyEncoder {
  // the first byte of an encoded key is used to indicate nullity
  static constexpr bool kExtraByteForNull = true;

  static constexpr uint8_t kNullByte = 1;
  static constexpr uint8_t kValidByte = 0;

  virtual ~KeyEncoder() = default;

  virtual void AddLength(const ExecValue& value, int64_t batch_length,
                         int32_t* lengths) = 0;

  virtual void AddLengthNull(int32_t* length) = 0;

  virtual Status Encode(const ExecValue&, int64_t batch_length,
                        uint8_t** encoded_bytes) = 0;

  virtual void EncodeNull(uint8_t** encoded_bytes) = 0;

  virtual Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes,
                                                    int32_t length, MemoryPool*) = 0;

  // extract the null bitmap from the leading nullity bytes of encoded keys
  static Status DecodeNulls(MemoryPool* pool, int32_t length, uint8_t** encoded_bytes,
                            std::shared_ptr<Buffer>* null_bitmap, int32_t* null_count);

  static bool IsNull(const uint8_t* encoded_bytes) {
    return encoded_bytes[0] == kNullByte;
  }
};

struct BooleanKeyEncoder : KeyEncoder {
  static constexpr int kByteWidth = 1;

  void AddLength(const ExecValue& data, int64_t batch_length, int32_t* lengths) override;

  void AddLengthNull(int32_t* length) override;

  Status Encode(const ExecValue& data, int64_t batch_length,
                uint8_t** encoded_bytes) override;

  void EncodeNull(uint8_t** encoded_bytes) override;

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override;
};

struct FixedWidthKeyEncoder : KeyEncoder {
  explicit FixedWidthKeyEncoder(std::shared_ptr<DataType> type)
      : type_(std::move(type)),
        byte_width_(checked_cast<const FixedWidthType&>(*type_).bit_width() / 8) {}

  void AddLength(const ExecValue& data, int64_t batch_length, int32_t* lengths) override;

  void AddLengthNull(int32_t* length) override;

  Status Encode(const ExecValue& data, int64_t batch_length,
                uint8_t** encoded_bytes) override;

  void EncodeNull(uint8_t** encoded_bytes) override;

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override;

  std::shared_ptr<DataType> type_;
  int byte_width_;
};

struct DictionaryKeyEncoder : FixedWidthKeyEncoder {
  DictionaryKeyEncoder(std::shared_ptr<DataType> type, MemoryPool* pool)
      : FixedWidthKeyEncoder(std::move(type)), pool_(pool) {}

  Status Encode(const ExecValue& data, int64_t batch_length,
                uint8_t** encoded_bytes) override;

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override;

  MemoryPool* pool_;
  std::shared_ptr<Array> dictionary_;
};

// Appends the null_byte and size to encoded_ptr.
// encoded_ptr will now point to the address after the offset.
template<typename Offset>
void EncodeVarLengthHeader(uint8_t*& encoded_ptr, uint8_t null_byte, Offset size = 0) {
  *encoded_ptr++ = null_byte;
  util::SafeStore(encoded_ptr, static_cast<Offset>(size));
  encoded_ptr += sizeof(Offset);
}

// Encodes entire variable length element to encoded_ptr
// Appends the null_byte, size and the encoded value to encoded_ptr.
// encoded_ptr will now point to the address after the encoded value.
template<typename Offset>
void EncodeVarLengthElement(uint8_t*& encoded_ptr, uint8_t null_byte,
                            Offset size, std::string_view value) {
  EncodeVarLengthHeader<Offset>(encoded_ptr, null_byte, size);
  memcpy(encoded_ptr, value.data(), value.size());
  encoded_ptr += value.size();
}

template <typename T>
struct VarLengthKeyEncoder : KeyEncoder {
  using Offset = typename T::offset_type;

  void AddLength(const ExecValue& data, int64_t batch_length, int32_t* lengths) override {
    if (data.is_array()) {
      int64_t i = 0;
      VisitArraySpanInline<T>(
          data.array,
          [&](std::string_view bytes) {
            lengths[i++] +=
                kExtraByteForNull + sizeof(Offset) + static_cast<int32_t>(bytes.size());
          },
          [&] { lengths[i++] += kExtraByteForNull + sizeof(Offset); });
    } else {
      const Scalar& scalar = *data.scalar;
      const int32_t buffer_size =
          scalar.is_valid ? static_cast<int32_t>(UnboxScalar<T>::Unbox(scalar).size())
                          : 0;
      for (int64_t i = 0; i < batch_length; i++) {
        lengths[i] += kExtraByteForNull + sizeof(Offset) + buffer_size;
      }
    }
  }

  void AddLengthNull(int32_t* length) override {
    *length += kExtraByteForNull + sizeof(Offset);
  }

  Status Encode(const ExecValue& data, int64_t batch_length,
                uint8_t** encoded_bytes) override {
    if (data.is_array()) {
      VisitArraySpanInline<T>(
          data.array,
          [&](std::string_view bytes) {
            EncodeVarLengthElement<Offset>(*encoded_bytes++, kValidByte,
               bytes.size(), bytes);
          },
          [&] {
            EncodeVarLengthHeader<Offset>(*encoded_bytes++, kNullByte);
          });
    } else {
      const auto& scalar = data.scalar_as<BaseBinaryScalar>();
      if (scalar.is_valid) {
        const auto& bytes = std::string_view(
              reinterpret_cast<const char*>(scalar.value->data()), scalar.value->size());
        for (int64_t i = 0; i < batch_length; i++) {
          EncodeVarLengthElement<Offset>(*encoded_bytes++, kValidByte,
            bytes.size(), bytes);
        }
      } else {
        for (int64_t i = 0; i < batch_length; i++) {
          EncodeVarLengthHeader<Offset>(*encoded_bytes++, kNullByte);
        }
      }
    }
    return Status::OK();
  }

  void EncodeNull(uint8_t** encoded_bytes) override {
    EncodeVarLengthHeader<Offset>(*encoded_bytes, kNullByte);
  }

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override {
    std::shared_ptr<Buffer> null_buf;
    int32_t null_count;
    ARROW_RETURN_NOT_OK(DecodeNulls(pool, length, encoded_bytes, &null_buf, &null_count));

    Offset length_sum = 0;
    for (int32_t i = 0; i < length; ++i) {
      length_sum += util::SafeLoadAs<Offset>(encoded_bytes[i]);
    }

    ARROW_ASSIGN_OR_RAISE(auto offset_buf,
                          AllocateBuffer(sizeof(Offset) * (1 + length), pool));
    ARROW_ASSIGN_OR_RAISE(auto key_buf, AllocateBuffer(length_sum));

    auto raw_offsets = reinterpret_cast<Offset*>(offset_buf->mutable_data());
    auto raw_keys = key_buf->mutable_data();

    Offset current_offset = 0;
    for (int32_t i = 0; i < length; ++i) {
      raw_offsets[i] = current_offset;

      auto key_length = util::SafeLoadAs<Offset>(encoded_bytes[i]);
      encoded_bytes[i] += sizeof(Offset);

      memcpy(raw_keys + current_offset, encoded_bytes[i], key_length);
      encoded_bytes[i] += key_length;

      current_offset += key_length;
    }
    raw_offsets[length] = current_offset;

    return ArrayData::Make(
        type_, length, {std::move(null_buf), std::move(offset_buf), std::move(key_buf)},
        null_count);
  }

  explicit VarLengthKeyEncoder(std::shared_ptr<DataType> type) : type_(std::move(type)) {}

  std::shared_ptr<DataType> type_;
};

// Iterates over an ArraySpan of lists and runs NullFunc if the list is null and ValidFunc
// otherwise.
// The arguments that ValidFunc takes in
// - starting offset of the list's nested-type
// - ending offset of the list's nested-type
// - the start of the validity buffer of the nested-type
// - the start of the nested-type's buffers[1]
template <typename T, typename ValidFunc, typename NullFunc = std::function<void()>>
void IterateListArray(const ArraySpan& arr, ValidFunc&& valid_func,
                      NullFunc&& null_func = NullFunc{}) {
  using offset_type = typename T::offset_type;
  constexpr uint8_t empty_value = 0;

  if (arr.length == 0) {
    return;
  }
  const offset_type* offsets = arr.GetValues<offset_type>(1);
  const uint8_t* child_data_buffer;
  if (arr.child_data[0].buffers[1].data == NULLPTR) {
    child_data_buffer = &empty_value;
  } else {
    child_data_buffer = arr.child_data[0].GetValues<uint8_t>(1);
  }

  const uint8_t* child_validity_buffer = arr.child_data[0].buffers[0].data == NULLPTR ?
    NULLPTR : arr.child_data[0].GetValues<uint8_t>(0);
  VisitBitBlocksVoid(
      arr.buffers[0].data, arr.offset, arr.length,
      [&](int64_t i) {
        valid_func(offsets[i], offsets[i + 1], child_validity_buffer, child_data_buffer);
      },
      std::forward<NullFunc>(null_func));
}

// Helps process a list scalar by running NullFunc if the scalar is null and ValidFunc
// otherwise.
// The arguments that ValidFunc takes in
// - the size of the scalar list
// - the start of the validity buffer of the nested-type
// - the start of the nested-type's buffers[1]
template<typename T, typename ValidFunc, typename NullFunc = std::function<void()>,
        typename ScalarType = typename TypeTraits<T>::ScalarType>
void ProcessListScalar(const ScalarType& scalar, ValidFunc&& valid_func,
                      NullFunc&& null_func = NullFunc{}) {
  if (scalar.is_valid) {
    std::shared_ptr<ArrayData> scalar_data = scalar.value->data();
    int64_t list_length = scalar_data->length;
    const uint8_t* scalar_validity_buf = scalar_data->buffers[0]->data() == NULLPTR ?
      NULLPTR : scalar_data->buffers[0]->data();
    const uint8_t* scalar_data_buf = scalar_data->buffers[1]->data();

    valid_func(list_length, scalar_validity_buf, scalar_data_buf);
  } else { // data.scalar->is_valid == false
    null_func();
  }
}

template<typename T>
struct ListFixedWidthChildEncoder : KeyEncoder {
  using Offset = typename T::offset_type;
  using ScalarType = typename TypeTraits<T>::ScalarType;

  explicit ListFixedWidthChildEncoder(std::shared_ptr<DataType> type) :
    type_(std::move(type)) {}

  void AddLength(const ExecValue& data, int64_t batch_length, int32_t* lengths) override {
    std::shared_ptr<DataType> child_type =
      std::static_pointer_cast<T>(type_)->value_type();
    const int32_t child_width = child_type->byte_width();

    if (data.is_array()) {
      int64_t i = 0;
      IterateListArray<T>(
        data.array,
        [&](Offset list_begin, Offset list_end, const uint8_t*, const uint8_t*) {
          Offset list_length = list_end - list_begin;
          lengths[i++] += kExtraByteForNull + sizeof(Offset) +
            list_length * (child_width + 1);
        },
        [&] { lengths[i++] += kExtraByteForNull + sizeof(Offset); });
    } else {
      int32_t buffer_size = 0;
      ProcessListScalar<T>(
        data.scalar_as<ScalarType>(),
        [&](int64_t list_length, const uint8_t*, const uint8_t*) {
          buffer_size = list_length * (child_width + 1);
        });
      for (int64_t i = 0; i < batch_length; i++) {
        lengths[i] += kExtraByteForNull + sizeof(Offset) + buffer_size;
      }
    }
  }

  void AddLengthNull(int32_t* length) override {
    *length += kExtraByteForNull + sizeof(Offset);
  }

  Status Encode(const ExecValue& data, int64_t batch_length,
                uint8_t** encoded_bytes) override {
    std::shared_ptr<DataType> child_type =
      std::static_pointer_cast<T>(type_)->value_type();
    const int32_t child_width = child_type->byte_width();

    auto encodeList = [&](Offset list_begin, Offset list_end,
      const uint8_t* list_validity_buffer, const uint8_t* list_data_buffer,
      uint8_t* encoded_ptr) {
      for (Offset j = list_begin; j < list_end; j++) {
        if (list_validity_buffer == NULLPTR ||
          bit_util::GetBit(list_validity_buffer, j)) {
          *encoded_ptr++ = kValidByte;
          memcpy(encoded_ptr, list_data_buffer + j * child_width, child_width);
        } else {
          *encoded_ptr++ = kNullByte;
          memset(encoded_ptr, 0, child_width);
        }
        encoded_ptr += child_width;
      }
      return (list_end - list_begin) * (child_width + 1);
    };

    if (data.is_array()) {
      IterateListArray<T>(
        data.array,
        [&](Offset list_begin, Offset list_end,
          const uint8_t* list_validity_buffer, const uint8_t* list_data_buffer) {
          Offset list_length = list_end - list_begin;
          EncodeVarLengthHeader<Offset>(*encoded_bytes, kValidByte, list_length);
          *encoded_bytes += encodeList(
            list_begin, list_end, list_validity_buffer, list_data_buffer, *encoded_bytes);
          encoded_bytes++;
        },
        [&] {
          EncodeVarLengthHeader<Offset>(*encoded_bytes++, kNullByte);
        });
    } else {
      ProcessListScalar<T>(
        data.scalar_as<ScalarType>(),
        [&](int64_t list_length,
          const uint8_t* scalar_validity_buf, const uint8_t* scalar_data_buf) {
          int64_t buffer_size = list_length * (child_width + 1);
          std::unique_ptr<uint8_t[]> raw_scalar_value =
            std::make_unique<uint8_t[]>(buffer_size);
          encodeList(0, list_length, scalar_validity_buf, scalar_data_buf,
            raw_scalar_value.get());
          std::string_view raw_value_str
            (reinterpret_cast<char *>(raw_scalar_value.get()), buffer_size);

          for (int64_t i = 0; i < batch_length; i++) {
            EncodeVarLengthElement<Offset>(*encoded_bytes++, kValidByte,
              list_length, raw_value_str);
          }
        },
        [&]() {
          for (int64_t i = 0; i < batch_length; i++) {
            EncodeVarLengthHeader<Offset>(*encoded_bytes++, kNullByte);
          }
        });
    }
    return Status::OK();
  }

  void EncodeNull(uint8_t** encoded_bytes) override {
    EncodeVarLengthHeader<Offset>(*encoded_bytes++, kNullByte);
  }

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override {
    std::shared_ptr<DataType> child_type =
      std::static_pointer_cast<T>(type_)->value_type();
    int32_t child_width = child_type->byte_width();

    std::shared_ptr<Buffer> null_buf;
    int32_t null_count;
    ARROW_RETURN_NOT_OK(DecodeNulls(pool, length, encoded_bytes, &null_buf, &null_count));

    int64_t length_sum = 0;
    for (int64_t i = 0; i < length; ++i) {
      length_sum += util::SafeLoadAs<Offset>(encoded_bytes[i]);
    }

    ARROW_ASSIGN_OR_RAISE(auto offset_buf,
                          AllocateBuffer(sizeof(Offset) * (1 + length), pool));
    ARROW_ASSIGN_OR_RAISE(auto child_value_buf,
                          AllocateBuffer(length_sum * child_width, pool));
    ARROW_ASSIGN_OR_RAISE(auto child_validity_buf, AllocateBitmap(length_sum, pool));

    auto raw_offsets = reinterpret_cast<Offset*>(offset_buf->mutable_data());
    auto raw_values = child_value_buf->mutable_data();
    auto raw_child_validity = child_validity_buf->mutable_data();

    FirstTimeBitmapWriter writer(raw_child_validity, 0, length_sum);
    Offset current_child_offset = 0;
    int64_t child_null_count = 0;
    for (int64_t i = 0; i < length; ++i) {
      raw_offsets[i] = current_child_offset;

      auto list_length = util::SafeLoadAs<Offset>(encoded_bytes[i]);
      encoded_bytes[i] += sizeof(Offset);

      for (int64_t j = 0; j < list_length; j++) {
        if (encoded_bytes[i][0] == kValidByte) {
          writer.Set();
          encoded_bytes[i] += kExtraByteForNull;
          memcpy(raw_values + (current_child_offset + j) * child_width,
            encoded_bytes[i], child_width);
        } else { // encoded_bytes[i][0] == kNullByte
          writer.Clear();
          child_null_count++;
          encoded_bytes[i] += kExtraByteForNull;
          memset(raw_values + (current_child_offset + j) * child_width, 0, child_width);
        }
        encoded_bytes[i] += child_width;
        writer.Next();
      }
      current_child_offset += list_length;
    }
    raw_offsets[length] = current_child_offset;
    writer.Finish();

    std::vector<std::shared_ptr<ArrayData>> child_data = {ArrayData::Make(
      std::static_pointer_cast<T>(type_)->value_type(), length_sum,
        {std::move(child_validity_buf), std::move(child_value_buf)}, child_null_count)};

    return ArrayData::Make(
      type_, length, {std::move(null_buf), std::move(offset_buf)},
        child_data, null_count);
  }

  std::shared_ptr<DataType> type_;
};

struct NullKeyEncoder : KeyEncoder {
  void AddLength(const ExecValue&, int64_t batch_length, int32_t* lengths) override {}

  void AddLengthNull(int32_t* length) override {}

  Status Encode(const ExecValue& data, int64_t batch_length,
                uint8_t** encoded_bytes) override {
    return Status::OK();
  }

  void EncodeNull(uint8_t** encoded_bytes) override {}

  Result<std::shared_ptr<ArrayData>> Decode(uint8_t** encoded_bytes, int32_t length,
                                            MemoryPool* pool) override {
    return ArrayData::Make(null(), length, {NULLPTR}, length);
  }
};

class ARROW_EXPORT RowEncoder {
 public:
  static constexpr int kRowIdForNulls() { return -1; }

  void Init(const std::vector<TypeHolder>& column_types, ExecContext* ctx);
  void Clear();
  Status EncodeAndAppend(const ExecSpan& batch);
  Result<ExecBatch> Decode(int64_t num_rows, const int32_t* row_ids);

  inline std::string encoded_row(int32_t i) const {
    if (i == kRowIdForNulls()) {
      return std::string(reinterpret_cast<const char*>(encoded_nulls_.data()),
                         encoded_nulls_.size());
    }
    int32_t row_length = offsets_[i + 1] - offsets_[i];
    return std::string(reinterpret_cast<const char*>(bytes_.data() + offsets_[i]),
                       row_length);
  }

  int32_t num_rows() const {
    return offsets_.size() == 0 ? 0 : static_cast<int32_t>(offsets_.size() - 1);
  }

 private:
  ExecContext* ctx_;
  std::vector<std::shared_ptr<KeyEncoder>> encoders_;
  std::vector<int32_t> offsets_;
  std::vector<uint8_t> bytes_;
  std::vector<uint8_t> encoded_nulls_;
  std::vector<std::shared_ptr<ExtensionType>> extension_types_;
};

}  // namespace internal
}  // namespace compute
}  // namespace arrow
