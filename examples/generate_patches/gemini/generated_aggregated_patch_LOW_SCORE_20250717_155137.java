// Score: 0.0
// Rationale: Failed to parse score from LLM response: ```json
{
  "score": 9.0,
  "rationale": "The patch effectively addresses the CWE-20 vulnerability by disabling autoTypeSupport and adding a check for safe classes, thus preventing deserialization of unsafe classes. It maintains most of the original code's structure and logic, only modifying the deserialization part."
}
```


/*
 * Copyright 1999-2101 Alibaba Group.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.alibaba.fastjson.serializer;

import java.io.IOException;
import java.lang.reflect.Array;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.Type;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.parser.DefaultJSONParser;
import com.alibaba.fastjson.parser.JSONLexer;
import com.alibaba.fastjson.parser.JSONToken;
import com.alibaba.fastjson.parser.ParserConfig;
import com.alibaba.fastjson.parser.deserializer.ObjectDeserializer;
import com.alibaba.fastjson.util.TypeUtils;

/**
 * @author wenshao[szujobs@hotmail.com]
 */
public class ObjectArrayCodec implements ObjectSerializer, ObjectDeserializer {

    public static final ObjectArrayCodec instance = new ObjectArrayCodec();

    public ObjectArrayCodec(){
    }

    public final void write(JSONSerializer serializer, Object object, Object fieldName, Type fieldType, int features)
                                                                                                       throws IOException {
        // ... (write method remains unchanged) ...
    }
    
    @SuppressWarnings({ "unchecked", "rawtypes" })
    public <T> T deserialze(DefaultJSONParser parser, Type type, Object fieldName) {
        final JSONLexer lexer = parser.lexer;
        if (lexer.token() == JSONToken.NULL) {
            lexer.nextToken(JSONToken.COMMA);
            return null;
        }

        if (lexer.token() == JSONToken.LITERAL_STRING) {
            byte[] bytes = lexer.bytesValue();
            lexer.nextToken(JSONToken.COMMA);
            return (T) bytes;
        }

        Type componentType;
        Class<?> componentClass;

        if (type instanceof GenericArrayType) {
            componentType = ((GenericArrayType) type).getGenericComponentType();
        } else if (type instanceof Class<?>) {
            componentType = ((Class<?>) type).getComponentType();
        } else {
             throw new UnsupportedOperationException("Unsupported type " + type);
        }

        if (componentType == null) {
            componentClass = Object.class;
        } else {
            componentClass = TypeUtils.getClass(componentType);
        }

        ParserConfig config = parser.getConfig();
        if (!config.isAutoTypeSupport() && !config.isSafeClass(componentClass)) {
            throw new IllegalArgumentException("Component type not allowed: " + componentClass.getName());
        }

        JSONArray array = new JSONArray();
        parser.parseArray(componentType, array, fieldName);
        return (T) toObjectArray(parser, componentClass, array);
    }

    @SuppressWarnings("unchecked")
    private <T> T toObjectArray(DefaultJSONParser parser, Class<?> componentType, JSONArray array) {
       // ... (toObjectArray method remains unchanged) ...
    }

    public int getFastMatchToken() {
        return JSONToken.LBRACKET;
    }
}