#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <base/string.h>
#include <base/io.h>
#include <base/vector.h>

#include "mlir_parser.h"
#include "tokenizer.h"
#include "op_parsers.h"

// Helper function declarations
static bool parse_type_string(Parser *parser, string *out) {
    if (!(parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION))) {
        return false;
    }

    string ty = parser_token_str(parser);
    parser_next_token(parser);

    // Dialect types starting with '!'
    if (ty.size == 1 && ty.str[0] == '!' && (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME))) {
        ty = str_concat(parser->arena, ty, parser_token_str(parser));
        parser_next_token(parser);
    }

    // Optional angle-bracketed content (supports nesting)
    if (parser_peek(parser, TK_LANGLE)) {
        int depth = 0;
        do {
            if (parser_peek(parser, TK_LANGLE)) depth++;
            else if (parser_peek(parser, TK_RANGLE)) depth--;
            ty = str_concat(parser->arena, ty, parser_token_str(parser));
            parser_next_token(parser);
        } while (depth > 0 && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
    }

    *out = ty;
    return true;
}

// Helper function to consume optional hash selector
void consume_optional_hash_selector(Parser *parser) {
    // Consume tokens like #0, #1 after a register (e.g., %49#0) and ignore them
    if (parser_peek(parser, TK_HASH_NAME)) {
        parser_next_token(parser);
    }
}

// Include the extracted functions
void parse_generic_attrs_and_result_type(Parser *parser, Operation *op) {
    // Attributes block
    // Handle generic attributes in angle-bracket form: <{ ... }>
    if (parser_peek(parser, TK_LANGLE)) {
        // Lookahead for '<{' sequence
        uint64_t save_first = parser->first, save_last = parser->last, save_cur = parser->cur; TokenType save_sym = parser->sym;
        parser_expect(parser, TK_LANGLE);
        bool had_attrs=false;
        if (parser_peek(parser, TK_LBRACE)) {
            parser_expect(parser, TK_LBRACE);
            Attribute **attrs = NULL; size_t n_attrs=0; size_t cap=4; attrs = arena_alloc_array(parser->arena, Attribute*, cap);
            while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                    string attr_name = parser_token_str(parser); parser_next_token(parser);
                    if (parser_peek(parser, TK_EQUAL)) { parser_expect(parser, TK_EQUAL);
                        if (n_attrs>=cap){ cap*=2; Attribute**na = arena_alloc_array(parser->arena, Attribute*, cap); for(size_t i=0;i<n_attrs;i++) na[i]=attrs[i]; attrs=na; }
                        Attribute *attr = arena_alloc(parser->arena, Attribute); attr->name=attr_name;
                        string payload = str_lit(""); int angle=0; while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_COMMA) && !parser_peek(parser, TK_RBRACE)) { string tok=parser_token_str(parser); if (parser_peek(parser, TK_LANGLE)) angle++; else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--; payload = payload.size ? str_concat(parser->arena, payload, tok) : tok; parser_next_token(parser); if (angle==0 && (parser_peek(parser, TK_COMMA) || parser_peek(parser, TK_RBRACE))) break; }
                        attr->kind = ATTR_KIND_STRING; attr->data.string_value = payload; attrs[n_attrs++] = attr; had_attrs=true;
                    }
                    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                } else { parser_next_token(parser); }
            }
            if (parser_peek(parser, TK_RBRACE)) parser_expect(parser, TK_RBRACE);
            if (had_attrs) { op->attributes = attrs; op->n_attributes = n_attrs; }
            if (parser_peek(parser, TK_RANGLE)) parser_expect(parser, TK_RANGLE);
        } else {
            // Not actually an attribute block; rewind
            parser->first = save_first; parser->last = save_last; parser->cur = save_cur; parser->sym = save_sym;
        }
    }
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        
        Attribute **attrs = NULL;
        size_t n_attrs = 0;
        size_t attrs_capacity = 4;
        attrs = arena_alloc_array(parser->arena, Attribute*, attrs_capacity);
        
        while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
            // Parse attribute name
            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                string attr_name = parser_token_str(parser);
                parser_next_token(parser);
                
                // Expect '='
                if (parser_peek(parser, TK_EQUAL)) {
                    parser_expect(parser, TK_EQUAL);
                    
                    // Grow array if needed
                    if (n_attrs >= attrs_capacity) {
                        attrs_capacity *= 2;
                        Attribute **new_attrs = arena_alloc_array(parser->arena, Attribute*, attrs_capacity);
                        for (size_t i = 0; i < n_attrs; i++) new_attrs[i] = attrs[i];
                        attrs = new_attrs;
                    }
                    
                    Attribute *attr = arena_alloc(parser->arena, Attribute);
                    attr->name = attr_name;
                    
                    // Parse attribute value: capture complex payload verbatim until ',' or '}'
                    string payload = str_lit("");
                    int angle = 0;
                    while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_COMMA) && !parser_peek(parser, TK_RBRACE)) {
                        string tok = parser_token_str(parser);
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                        payload = payload.size ? str_concat(parser->arena, payload, tok) : tok;
                        parser_next_token(parser);
                        if (angle==0 && (parser_peek(parser, TK_COMMA) || parser_peek(parser, TK_RBRACE))) break;
                    }
                    attr->kind = ATTR_KIND_STRING;
                    attr->data.string_value = payload;
                    
                    attrs[n_attrs++] = attr;
                }
                
                // Skip comma if present
                if (parser_peek(parser, TK_COMMA)) {
                    parser_expect(parser, TK_COMMA);
                }
            } else {
                // Skip unknown token
                parser_next_token(parser);
            }
        }
        
        if (parser_peek(parser, TK_RBRACE)) {
            parser_expect(parser, TK_RBRACE);
        }
        
        // Store parsed attributes
        if (n_attrs > 0) {
            op->attributes = attrs;
            op->n_attributes = n_attrs;
        }
    }

    // Optional result type after ':'
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Handle forms like ": (type, ...) -> type"
        if (parser_peek(parser, TK_LPAREN)) {
            // Parse and capture the first type inside parentheses for classic printing
            parser_expect(parser, TK_LPAREN);
            string src_sig = str_lit("");
            (void)parse_type_string(parser, &src_sig);
            // Consume until matching ')'
            int depth = 1;
            while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            }

            // Optional arrow and a result type
            if (parser_peek(parser, TK_ARROW)) {
                parser_expect(parser, TK_ARROW);
            }

            string type_str = str_lit("");
            if (parse_type_string(parser, &type_str)) {
                op->n_result_types = 1;
                op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                op->result_types[0] = parse_type_from_string(parser->arena, type_str);
            }

            // Record that the signature used parenthesized operand types
            size_t n = op->n_attributes;
            Attribute **attrs = op->attributes;
            Attribute *flag = arena_alloc(parser->arena, Attribute);
            flag->kind = ATTR_KIND_BOOL;
            flag->name = str_lit("_sig_parens");
            flag->data.bool_value = true;
            if (attrs == NULL) {
                attrs = arena_alloc_array(parser->arena, Attribute*, 2);
                attrs[0] = flag; 
                // also store the src type string
                Attribute *srca = arena_alloc(parser->arena, Attribute);
                srca->kind = ATTR_KIND_STRING; srca->name = str_lit("_sig_src"); srca->data.string_value = src_sig;
                attrs[1] = srca; n = 2;
            } else {
                Attribute **new_attrs = arena_alloc_array(parser->arena, Attribute*, n+2);
                for (size_t i = 0; i < n; i++) new_attrs[i] = attrs[i];
                new_attrs[n] = flag; 
                Attribute *srca = arena_alloc(parser->arena, Attribute);
                srca->kind = ATTR_KIND_STRING; srca->name = str_lit("_sig_src"); srca->data.string_value = src_sig;
                new_attrs[n+1] = srca;
                n = n+2; attrs = new_attrs;
            }
            op->attributes = attrs;
            op->n_attributes = n;

            // Optional trailing loc()
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                op->location = parse_loc(parser);
            }
            return;
        }

        if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
            // Parse a type token (could be src type or result type)
            string type_left = str_lit("");
            parse_type_string(parser, &type_left);

            // Handle special form: "type * type -> result"
            if (parser_peek(parser, TK_STAR)) {
                parser_expect(parser, TK_STAR);
                string type_right_mul = str_lit("");
                parse_type_string(parser, &type_right_mul);
                // Expect arrow and a result type
                if (parser_peek(parser, TK_ARROW)) parser_expect(parser, TK_ARROW);
                string type_res = str_lit("");
                if (parse_type_string(parser, &type_res)) {
                    op->n_result_types = 1;
                    op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                    op->result_types[0] = parse_type_from_string(parser->arena, type_res);
                }
                // Done handling this signature. Do not consume further (keep trailing loc()).
                return;
            } else if (parser_peek(parser, TK_ARROW)) {
                // Form ": <src-type> -> <result-type>"
                parser_expect(parser, TK_ARROW);
                string type_right = str_lit("");
                if (parse_type_string(parser, &type_right)) {
                    op->n_result_types = 1;
                    op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                    op->result_types[0] = parse_type_from_string(parser->arena, type_right);
                }
                // Record source signature string for classic printing
                size_t n = op->n_attributes;
                Attribute **attrs = op->attributes;
                Attribute *srca = arena_alloc(parser->arena, Attribute);
                srca->kind = ATTR_KIND_STRING; srca->name = str_lit("_sig_src"); srca->data.string_value = type_left;
                if (attrs == NULL) {
                    attrs = arena_alloc_array(parser->arena, Attribute*, 1); attrs[0] = srca; n = 1;
                } else {
                    Attribute **new_attrs = arena_alloc_array(parser->arena, Attribute*, n+1);
                    for (size_t i = 0; i < n; i++) new_attrs[i] = attrs[i];
                    new_attrs[n] = srca; n = n+1; attrs = new_attrs;
                }
                op->attributes = attrs; op->n_attributes = n;
            } else if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
                // Casts like ": src_ty to dst_ty"; set result to dst_ty
                parser_expect(parser, TK_NAME);
                string type_dst = str_lit("");
                if (parse_type_string(parser, &type_dst)) {
                    op->n_result_types = 1;
                    op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                    op->result_types[0] = parse_type_from_string(parser->arena, type_dst);
                }
            } else if (parser_peek(parser, TK_COMMA)) {
                // Operand type list ": type, type, ..." — consume conservatively
                while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                    parser_next_token(parser);
                }
            } else {
                // Treat as single result type ": type" for most ops,
                // but do NOT override for compare ops like arith.cmpi where
                // the colon type is the operand type, not the result.
                if (!(op->opname.size > 0 && str_eq(op->opname, str_lit("arith.cmpi")))) {
                    op->n_result_types = 1;
                    op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                    op->result_types[0] = arena_alloc(parser->arena, Type);
                    op->result_types[0] = parse_type_from_string(parser->arena, type_left);
                }
            }
        }
    }

    // Or optional result type directly after operands via '-> type'
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        string type_str = str_lit("");
        if (parse_type_string(parser, &type_str)) {
            op->n_result_types = 1;
            op->result_types = arena_alloc_array(parser->arena, Type*, 1);
            op->result_types[0] = parse_type_from_string(parser->arena, type_str);
        }
    }

    // Optional trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
}

void parse_arith_constant(Parser *parser, Operation *op) {
    // arith.constant value : type
    if (parser_peek(parser, TK_INTEGER)) {
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_INTEGER);

        op->n_attributes = 1;
        op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
        op->attributes[0] = arena_alloc(parser->arena, Attribute);
        op->attributes[0]->kind = ATTR_KIND_INTEGER;

        int64_t parsed_value = 0;
        for (size_t i = 0; i < value_str.size; i++) {
            if (value_str.str[i] >= '0' && value_str.str[i] <= '9') {
                parsed_value = parsed_value * 10 + (value_str.str[i] - '0');
            }
        }
        op->attributes[0]->data.integer_value = parsed_value;
        op->attributes[0]->name = str_lit("value");
    } else if (parser_peek(parser, TK_REAL)) {
        // Handle floating point constants like 0.000000e+00
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_REAL);

        op->n_attributes = 1;
        op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
        op->attributes[0] = arena_alloc(parser->arena, Attribute);
        op->attributes[0]->kind = ATTR_KIND_FLOAT;

        // Parse floating point value using strtod
        char *str_copy = arena_alloc_array(parser->arena, char, value_str.size + 1);
        memcpy(str_copy, value_str.str, value_str.size);
        str_copy[value_str.size] = '\0';
        double parsed_value = strtod(str_copy, NULL);
        op->attributes[0]->data.float_value = parsed_value;
        op->attributes[0]->name = str_lit("value");
    } else if (parser_peek(parser, TK_NAME)) {
        string name_str = parser_token_str(parser);
        // Handle boolean constants: true or false
        if (str_eq(name_str, str_lit("true")) || str_eq(name_str, str_lit("false"))) {
            parser_expect(parser, TK_NAME);

            op->n_attributes = 1;
            op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
            op->attributes[0] = arena_alloc(parser->arena, Attribute);
            op->attributes[0]->kind = ATTR_KIND_INTEGER;
            op->attributes[0]->data.integer_value = str_eq(name_str, str_lit("true")) ? 1 : 0;
            op->attributes[0]->name = str_lit("value");

            // Boolean constants have implicit i1 type
            op->n_result_types = 1;
            op->result_types = arena_alloc_array(parser->arena, Type*, 1);
            op->result_types[0] = parse_type_from_string(parser->arena, str_lit("i1"));
            return;
        } else {
            // Capture payload like dense<...> verbatim until ':'
            string payload = str_lit("");
            int angle = 0;
            do {
                string tok = parser_token_str(parser);
                payload = payload.size ? str_concat(parser->arena, payload, tok) : tok;
                parser_next_token(parser);
                if (parser_peek(parser, TK_LANGLE)) angle++;
                else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
                if (angle == 0 && parser_peek(parser, TK_COLON)) break;
            } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
            // Store as string attribute for classic printing
            op->n_attributes = 1;
            op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
            op->attributes[0] = arena_alloc(parser->arena, Attribute);
            op->attributes[0]->kind = ATTR_KIND_STRING;
            op->attributes[0]->name = str_lit("value_text");
            op->attributes[0]->data.string_value = payload;
        }
    } else {
        // Fallback: consume attribute-like payload (e.g., dense<...>) until ':' or EOL
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_COLON)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            parser_next_token(parser);
        }
    }
}

void parse_arith_binary(Parser *parser, Operation *op) {
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 2);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, operand);

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str2 = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *operand2 = symbol_table_lookup(&parser->symbol_table, reg_str2);
                if (!operand2) {
                    parser_warning(parser, format(parser->arena, str_lit("Undefined value: {}"), reg_str2), parser->first, parser->last);
                    operand2 = create_value_ref(parser->arena, BLOCK_ARG);
                    operand2->register_name = reg_str2;
                    operand2->type = arena_alloc(parser->arena, Type);
                    operand2->type = parse_type_from_string(parser->arena, str_lit("unknown"));
                }
                VecValueRef_push_back(parser->arena, &operands, operand2);
            }
        }
    }

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Optional default result type for specific ops when not provided
    if (op->n_result_types == 0 && op->opname.size > 0 && str_eq(op->opname, str_lit("arith.addf"))) {
        op->n_result_types = 1;
        op->result_types = arena_alloc_array(parser->arena, Type*, 1);
        op->result_types[0] = parse_type_from_string(parser->arena, str_lit("tensor<16xf32>"));

    }
}

void parse_func_call(Parser *parser, Operation *op) {
    // Parse call @function(%args) : (arg_types) -> result_type
    // Function name (@name is tokenized as TK_FUNCTION_NAME)
    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture callee
        string fname = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        // Store as attribute 'callee'
        op->n_attributes = 1;
        op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
        op->attributes[0] = arena_alloc(parser->arena, Attribute);
        op->attributes[0]->name = str_lit("callee");
        op->attributes[0]->kind = ATTR_KIND_STRING;
        op->attributes[0]->data.string_value = fname;
    }

    // Parse operands in parentheses
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);

        VecValueRef operands;
        VecValueRef_reserve(parser->arena, &operands, 4);

        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValueRef_push_back(parser->arena, &operands, operand);
            } else {
                parser_next_token(parser); // skip unknown tokens
            }

            if (parser_peek(parser, TK_COMMA)) {
                parser_expect(parser, TK_COMMA);
            } else if (!parser_peek(parser, TK_RPAREN)) {
                break;
            }
        }

        if (parser_peek(parser, TK_RPAREN)) {
            parser_expect(parser, TK_RPAREN);
        }

        op->operands = operands.data;
        op->n_operands = operands.size;
    }

    // Parse function signature: : (arg_types) -> result_type
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip arg types in parentheses
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int paren_depth = 1;
            while (paren_depth > 0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) paren_depth++;
                else if (parser_peek(parser, TK_RPAREN)) paren_depth--;
                parser_next_token(parser);
            }
        }

        // Parse arrow and result type
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);
            string type_str = str_lit("");
            if (parse_type_string(parser, &type_str)) {
                op->n_result_types = 1;
                op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                op->result_types[0] = parse_type_from_string(parser->arena, type_str);
            }
        }
    }
}

void parse_tt_get_program_id(Parser *parser, Operation *op) {
    if (parser_peek(parser, TK_NAME)) {
        string axis = parser_token_str(parser);
        parser_expect(parser, TK_NAME);

        op->n_attributes = 1;
        op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
        op->attributes[0] = arena_alloc(parser->arena, Attribute);
        op->attributes[0]->kind = ATTR_KIND_STRING;
        op->attributes[0]->data.string_value = axis;
        op->attributes[0]->name = str_lit("axis");
    }
    op->n_operands = 0;
    op->operands = NULL;
}

void parse_tt_splat(Parser *parser, Operation *op) {
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        op->n_operands = 1;
        op->operands = arena_alloc_array(parser->arena, ValueRef*, 1);
        op->operands[0] = operand;
    }
    // Infer result type if not already parsed from trailing type
    if (op->n_result_types == 0) {
        op->n_result_types = 1;
        op->result_types = arena_alloc_array(parser->arena, Type*, 1);
        op->result_types[0] = arena_alloc(parser->arena, Type);
        if (op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
            string operand_type = type_to_string(parser->arena, op->operands[0]->type);
            if (str_eq(operand_type, str_lit("!tt.ptr<f32>"))) {
                op->result_types[0] = parse_type_from_string(parser->arena, str_lit("tensor<16x!tt.ptr<f32>>"));
            } else if (str_eq(operand_type, str_lit("i32"))) {
                op->result_types[0] = parse_type_from_string(parser->arena, str_lit("tensor<16xi32>"));
            } else {
                op->result_types[0] = parse_type_from_string(parser->arena, str_lit("tensor<16xi32>"));
            }
        } else {
            op->result_types[0] = parse_type_from_string(parser->arena, str_lit("tensor<16xi32>"));
        }

    }
}

void parse_tt_make_range(Parser *parser, Operation *op) {
    op->n_operands = 0;
    op->operands = NULL;
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        op->n_attributes = 2;
        op->attributes = arena_alloc_array(parser->arena, Attribute*, 2);
        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME); // end
            parser_expect(parser, TK_EQUAL);
            // capture integer
            int64_t end_val = 0;
            if (parser_peek(parser, TK_INTEGER)) {
                string ival = parser_token_str(parser);
                for (size_t k = 0; k < ival.size; k++) {
                    char c = ival.str[k];
                    if (c >= '0' && c <= '9') end_val = end_val * 10 + (c - '0');
                }
                parser_expect(parser, TK_INTEGER);
            }
            // optional ': i32'
            if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                if (parser_peek(parser, TK_NAME)) parser_expect(parser, TK_NAME);
            }
            op->attributes[0] = arena_alloc(parser->arena, Attribute);
            op->attributes[0]->kind = ATTR_KIND_INTEGER;
            op->attributes[0]->name = str_lit("end");
            op->attributes[0]->data.integer_value = end_val;
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME); // start
            parser_expect(parser, TK_EQUAL);
            int64_t start_val = 0;
            if (parser_peek(parser, TK_INTEGER)) {
                string ival = parser_token_str(parser);
                for (size_t k = 0; k < ival.size; k++) {
                    char c = ival.str[k];
                    if (c >= '0' && c <= '9') start_val = start_val * 10 + (c - '0');
                }
                parser_expect(parser, TK_INTEGER);
            }
            if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                if (parser_peek(parser, TK_NAME)) parser_expect(parser, TK_NAME);
            }
            op->attributes[1] = arena_alloc(parser->arena, Attribute);
            op->attributes[1]->kind = ATTR_KIND_INTEGER;
            op->attributes[1]->name = str_lit("start");
            op->attributes[1]->data.integer_value = start_val;
        }
        parser_expect(parser, TK_RBRACE);
    }
    // Result type parsed by generic handler after ':'
}

void parse_tt_addptr_load_store(Parser *parser, Operation *op) {
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 2);
    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        consume_optional_hash_selector(parser);
        ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, operand);
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA); else break;
    }
    op->operands = operands.data;
    op->n_operands = operands.size;

    // For tt.addptr/tt.load, prefer explicit type after ':' when present; otherwise fall back.
    if (str_eq(op->opname, str_lit("tt.addptr")) || str_eq(op->opname, str_lit("tt.load"))) {
        if (parser_peek(parser, TK_COLON)) {
            parser_expect(parser, TK_COLON);

            // Parse first type token into a string (handles !dialect<...> nesting)
            string type_left = str_lit("");
            if (parse_type_string(parser, &type_left)) {
                // Set result type
                op->n_result_types = 1;
                op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                if (str_eq(op->opname, str_lit("tt.addptr"))) {
                    // For addptr, result type is the pointer tensor type
                    op->result_types[0] = parse_type_from_string(parser->arena, type_left);
                } else {
                    // For load, convert pointer tensor element to value element type (e.g., f32)
                    // Do a textual conversion: replace '!tt.ptr<...>' payload with element inside or f32
                    string s = type_left;
                    // Find '!tt.ptr<' inside s
                    size_t pos = SIZE_MAX;
                    for (size_t i = 0; i + 8 <= s.size; i++) {
                        if (s.str[i]=='!' && i+8<=s.size && s.str[i+1]=='t' && s.str[i+2]=='t' && s.str[i+3]=='.' && s.str[i+4]=='p' && s.str[i+5]=='t' && s.str[i+6]=='r' && s.str[i+7]=='<') { pos = i; break; }
                    }
                    if (pos != SIZE_MAX) {
                        // Extract element type inside '!tt.ptr<...>' before optional ','
                        size_t start = pos + 8;
                        size_t end = start;
                        while (end < s.size && s.str[end] != '>' && s.str[end] != ',') end++;
                        string elem = str_substr(s, start, end - start);
                        // Build new type string by replacing '!tt.ptr<...>' with 'elem'
                        string before = str_substr(s, 0, pos);
                        // Skip until closing '>'
                        size_t j = end;
                        while (j < s.size && s.str[j] != '>') j++;
                        if (j < s.size) j++; // include '>'
                        string after = str_substr(s, j, s.size - j);
                        string val_ty = before;
                        val_ty = str_concat(parser->arena, val_ty, elem.size ? elem : str_lit("f32"));
                        val_ty = str_concat(parser->arena, val_ty, after);
                        op->result_types[0] = parse_type_from_string(parser->arena, val_ty);
                    } else {
                        // Fallback: assume f32 element
                        op->result_types[0] = parse_type_from_string(parser->arena, s);
                    }
                }
            }
            // Consume remaining type list conservatively until end of list or loc
            if (parser_peek(parser, TK_COMMA)) {
                do {
                    parser_next_token(parser);
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE));
            }
        } else if (str_eq(op->opname, str_lit("tt.addptr")) && op->n_operands > 0 && op->operands[0] && op->operands[0]->type) {
            // Fallback heuristic when no explicit type list is present
            op->n_result_types = 1;
            op->result_types = arena_alloc_array(parser->arena, Type*, 1);
            op->result_types[0] = op->operands[0]->type;
        }
    }
}

void parse_tt_store(Parser *parser, Operation *op) {
    // Reuse common operand parsing for tt.addptr/load/store
    parse_tt_addptr_load_store(parser, op);

    // Parse attributes manually since tt.store doesn't produce results
    // Look for optional attribute dict: {key = value, ...}
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        
        Attribute **attrs = NULL;
        size_t n_attrs = 0;
        size_t attrs_capacity = 4;
        attrs = arena_alloc_array(parser->arena, Attribute*, attrs_capacity);
        
        while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
            // Parse attribute name
            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                string attr_name = parser_token_str(parser);
                parser_next_token(parser);
                
                // Expect '='
                if (parser_peek(parser, TK_EQUAL)) {
                    parser_expect(parser, TK_EQUAL);
                    
                    // Grow array if needed
                    if (n_attrs >= attrs_capacity) {
                        attrs_capacity *= 2;
                        Attribute **new_attrs = arena_alloc_array(parser->arena, Attribute*, attrs_capacity);
                        for (size_t i = 0; i < n_attrs; i++) {
                            new_attrs[i] = attrs[i];
                        }
                        attrs = new_attrs;
                    }
                    
                    // Create and parse attribute
                    Attribute *attr = arena_alloc(parser->arena, Attribute);
                    attr->name = attr_name;
                    
                    // Parse attribute value
                    if (parser_peek(parser, TK_INTEGER)) {
                        string value_str = parser_token_str(parser);
                        parser_expect(parser, TK_INTEGER);
                        attr->kind = ATTR_KIND_INTEGER;
                        attr->data.integer_value = atoll(value_str.str);
                    } else if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("false"))) {
                        parser_expect(parser, TK_NAME);
                        attr->kind = ATTR_KIND_BOOL;
                        attr->data.bool_value = false;
                    } else if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("true"))) {
                        parser_expect(parser, TK_NAME);
                        attr->kind = ATTR_KIND_BOOL;
                        attr->data.bool_value = true;
                    } else {
                        // Skip unknown attribute values
                        parser_next_token(parser);
                        attr->kind = ATTR_KIND_INTEGER;
                        attr->data.integer_value = 0;
                    }
                    
                    // Skip optional type annotation (: i32)
                    if (parser_peek(parser, TK_COLON)) {
                        parser_expect(parser, TK_COLON);
                        if (parser_peek(parser, TK_NAME)) {
                            parser_next_token(parser);
                        }
                    }
                    
                    attrs[n_attrs++] = attr;
                }
                
                // Skip optional comma
                if (parser_peek(parser, TK_COMMA)) {
                    parser_expect(parser, TK_COMMA);
                }
            } else {
                break;
            }
        }
        
        parser_expect(parser, TK_RBRACE);
        
        // Store attributes in operation
        op->attributes = attrs;
        op->n_attributes = n_attrs;
    }

    // Trailing operand type list ": ..." (consume conservatively)
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            // Allow loc() to be parsed by the generic helper
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
        // Optional trailing loc()
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }
}

void parse_tensor_extract(Parser *parser, Operation *op) {
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 2);
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *tensor_op = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!tensor_op) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, tensor_op);
    }
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                consume_optional_hash_selector(parser);
                consume_optional_hash_selector(parser);
                consume_optional_hash_selector(parser);
                consume_optional_hash_selector(parser);
                ValueRef *idx = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!idx) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValueRef_push_back(parser->arena, &operands, idx);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }
    op->operands = operands.data;
    op->n_operands = operands.size;
}

void parse_memref_load_or_store(Parser *parser, Operation *op) {
    VecValueRef operands; VecValueRef_reserve(parser->arena, &operands, 4);

    if (str_eq(op->opname, str_lit("memref.store"))) {
        // memref.store %value, %memref[indices] : memref<...>
        if (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            ValueRef *val = symbol_table_lookup(&parser->symbol_table, reg);
            if (!val) {
                val = create_value_ref(parser->arena, BLOCK_ARG);
                val->register_name = reg;
                val->type = arena_alloc(parser->arena, Type);
                val->type = parse_type_from_string(parser->arena, str_lit("unknown"));

            }
            VecValueRef_push_back(parser->arena, &operands, val);
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        if (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            ValueRef *val = symbol_table_lookup(&parser->symbol_table, reg);
            if (!val) {
                val = create_value_ref(parser->arena, BLOCK_ARG);
                val->register_name = reg;
                val->type = arena_alloc(parser->arena, Type);
                val->type = parse_type_from_string(parser->arena, str_lit("unknown"));

            }
            VecValueRef_push_back(parser->arena, &operands, val);
        }
    } else {
        // memref.load %memref[indices] : memref<...>
        if (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            ValueRef *val = symbol_table_lookup(&parser->symbol_table, reg);
            if (!val) {
                val = create_value_ref(parser->arena, BLOCK_ARG);
                val->register_name = reg;
                val->type = arena_alloc(parser->arena, Type);
                val->type = parse_type_from_string(parser->arena, str_lit("unknown"));

            }
            VecValueRef_push_back(parser->arena, &operands, val);
        }
    }

    // Optional index list in brackets
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        bool first = true;
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string idx = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *val = symbol_table_lookup(&parser->symbol_table, idx);
                if (!val) {
                    val = create_value_ref(parser->arena, BLOCK_ARG);
                    val->register_name = idx;
                    val->type = arena_alloc(parser->arena, Type);
                    val->type = parse_type_from_string(parser->arena, str_lit("index"));

                }
                VecValueRef_push_back(parser->arena, &operands, val);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    op->operands = operands.data;
    op->n_operands = operands.size;
}

void parse_memref_store(Parser *parser, Operation *op) {
    // Reuse common operand + index parsing
    parse_memref_load_or_store(parser, op);

    // Consume trailing ": memref<...>" conservatively and optional loc()
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }
    // Ensure no results
    op->n_result_types = 0;
}

void parse_vector_print(Parser *parser, Operation *op) {
    // Parse operands inside parentheses or as bare register(s)
    VecValueRef operands; VecValueRef_reserve(parser->arena, &operands, 2);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        bool first = true;
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *val = symbol_table_lookup(&parser->symbol_table, reg);
                if (!val) {
                    val = create_value_ref(parser->arena, BLOCK_ARG);
                    val->register_name = reg;
                    val->type = arena_alloc(parser->arena, Type);
                    val->type = parse_type_from_string(parser->arena, str_lit("unknown"));

                }
                VecValueRef_push_back(parser->arena, &operands, val);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    } else {
        // Bare operand form: vector.print %v : type
        while (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            ValueRef *val = symbol_table_lookup(&parser->symbol_table, reg);
            if (!val) {
                val = create_value_ref(parser->arena, BLOCK_ARG);
                val->register_name = reg;
                val->type = arena_alloc(parser->arena, Type);
                val->type = parse_type_from_string(parser->arena, str_lit("unknown"));

            }
            VecValueRef_push_back(parser->arena, &operands, val);
            if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA); else break;
        }
    }
    op->operands = operands.data;
    op->n_operands = operands.size;

    // Consume trailing ": type" and optional loc()
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }
    // Ensure no results
    op->n_result_types = 0;
}

void parse_std_constant(Parser *parser, Operation *op) {
    // Optional empty operand list in parentheses
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            parser_next_token(parser);
        }
        parser_expect(parser, TK_RPAREN);
    }
    // Optional attribute dict
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        int depth = 1;
        while (depth > 0 && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LBRACE)) depth++;
            else if (parser_peek(parser, TK_RBRACE)) depth--;
            parser_next_token(parser);
        }
    }
}

void parse_tt_reduce(Parser *parser, Operation *op) {
    // Parse "tt.reduce"(%operand) <{attributes}> ({region}) : (input_type) -> output_type
    
    // Parse operands: "tt.reduce"(%arg0)
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 1);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValueRef_push_back(parser->arena, &operands, operand);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }
    op->operands = operands.data;
    op->n_operands = operands.size;

    // Parse attributes: <{axis = 1 : i32}>
    if (parser_peek(parser, TK_LANGLE)) {
        parser_expect(parser, TK_LANGLE);
        if (parser_peek(parser, TK_LBRACE)) {
            parser_expect(parser, TK_LBRACE);
            // Parse axis attribute
            while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_NAME)) {
                    string attr_name = parser_token_str(parser);
                    parser_expect(parser, TK_NAME);
                    if (parser_peek(parser, TK_EQUAL)) {
                        parser_expect(parser, TK_EQUAL);
                        if (parser_peek(parser, TK_INTEGER)) {
                            string int_val = parser_token_str(parser);
                            parser_expect(parser, TK_INTEGER);
                            
                            // Store axis attribute
                            op->n_attributes = 1;
                            op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
                            op->attributes[0] = arena_alloc(parser->arena, Attribute);
                            op->attributes[0]->name = attr_name;
                            op->attributes[0]->kind = ATTR_KIND_INTEGER;
                            
                            // Convert string to integer
                            int64_t val = 0;
                            for (size_t i = 0; i < int_val.size; i++) {
                                char c = int_val.str[i];
                                if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
                            }
                            op->attributes[0]->data.integer_value = val;
                            
                            // Skip optional ': i32'
                            if (parser_peek(parser, TK_COLON)) {
                                parser_expect(parser, TK_COLON);
                                if (parser_peek(parser, TK_NAME)) parser_expect(parser, TK_NAME);
                            }
                        }
                    }
                } else {
                    parser_next_token(parser);
                }
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RBRACE);
        }
        parser_expect(parser, TK_RANGLE);
    }
    
    // Parse region: ({^bb0(%arg1: f32, %arg2: f32): ... })
    if (parser_peek(parser, TK_LPAREN_BRACE) || parser_peek(parser, TK_LPAREN)) {
        if (parser_peek(parser, TK_LPAREN_BRACE)) {
            parser_expect(parser, TK_LPAREN_BRACE);
        } else {
            parser_expect(parser, TK_LPAREN);
        }
        if (parser_peek(parser, TK_LBRACE_END)) {
            Region *region = parse_region(parser);
            op->n_regions = 1;
            op->regions = arena_alloc_array(parser->arena, Region*, 1);
            op->regions[0] = region;
        }
        parser_expect(parser, TK_RPAREN);
    }
    
    // Parse type signature: : (tensor<2x256xf32>) -> tensor<2xf32>
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        
        // Parse input type in parens
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            // Skip input type for now
            int paren_depth = 0;
            while (!parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) paren_depth++;
                else if (parser_peek(parser, TK_RPAREN)) {
                    if (paren_depth == 0) break;
                    paren_depth--;
                }
                parser_next_token(parser);
            }
            parser_expect(parser, TK_RPAREN);
        }
        
        // Parse arrow and result type
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);
            // Parse result type
            string type_str = str_lit("");
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_LBRACE_END)) {
                string tok = parser_token_str(parser);
                type_str = type_str.size ? str_concat(parser->arena, str_concat(parser->arena, type_str, str_lit(" ")), tok) : tok;
                parser_next_token(parser);
            }
            if (type_str.size > 0) {
                op->n_result_types = 1;
                op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                op->result_types[0] = parse_type_from_string(parser->arena, type_str);
            }
        }
    }
}

void parse_cf_br(Parser *parser, Operation *op) {
    // Parse cf.br ^bbX(%args : types)
    string target = str_lit("^bb1");
    if (parser_peek(parser, TK_CARET_NAME)) { target = parser_token_str(parser); parser_expect(parser, TK_CARET_NAME); }

    VecValueRef branch_args; VecValueRef_reserve(parser->arena, &branch_args, 4);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string vr = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *v = symbol_table_lookup(&parser->symbol_table, vr);
                if (v) VecValueRef_push_back(parser->arena, &branch_args, v);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                // Skip types list
                int angle=0;
                while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) { if (parser_peek(parser, TK_LANGLE)) angle++; else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--; parser_next_token(parser);}                
            } else { parser_next_token(parser); }
        }
        parser_expect(parser, TK_RPAREN);
    }
    op->operands = branch_args.data; op->n_operands = branch_args.size;
    // Store target as private attribute
    op->n_attributes = 1; op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
    op->attributes[0] = arena_alloc(parser->arena, Attribute);
    op->attributes[0]->name = str_lit("_target"); op->attributes[0]->kind = ATTR_KIND_STRING; op->attributes[0]->data.string_value = target;
    // trailing loc
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) op->location = parse_loc(parser);
}

void parse_cf_cond_br(Parser *parser, Operation *op) {
    // Parse: cf.cond_br %cond, ^bbX[(args : types)], ^bbY[(args : types)] [loc]
    VecValueRef operands; VecValueRef_reserve(parser->arena, &operands, 1);
    // condition
    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *cond = symbol_table_lookup(&parser->symbol_table, reg);
        if (cond) VecValueRef_push_back(parser->arena, &operands, cond);
    }
    // commas and targets
    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
    string ttrue = str_lit("");
    string tfalse = str_lit("");
    if (parser_peek(parser, TK_CARET_NAME)) {
        ttrue = parser_token_str(parser); parser_expect(parser, TK_CARET_NAME);
        // Optional argument list for true target: (^bbX(%v1, %v2 : ty1, ty2))
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int64_t ntrue = 0;
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string vr = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    ValueRef *v = symbol_table_lookup(&parser->symbol_table, vr);
                    if (v) VecValueRef_push_back(parser->arena, &operands, v);
                    ntrue++;
                    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                } else if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    // consume type list until ')'
                    int angle=0;
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                        parser_next_token(parser);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
            // store count for true branch
            // append attribute _ntrue (accumulate, don't overwrite later)
            size_t n = op->n_attributes; Attribute **attrs = op->attributes;
            Attribute *an = arena_alloc(parser->arena, Attribute);
            an->name = str_lit("_ntrue"); an->kind = ATTR_KIND_INTEGER; an->data.integer_value = ntrue;
            Attribute **na = arena_alloc_array(parser->arena, Attribute*, n+1);
            for (size_t i=0;i<n;i++) na[i]=attrs[i];
            na[n]=an;
            op->attributes=na; op->n_attributes=(int)(n+1);
        }
    }
    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
    if (parser_peek(parser, TK_CARET_NAME)) {
        tfalse = parser_token_str(parser); parser_expect(parser, TK_CARET_NAME);
        // Optional argument list for false target
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int64_t nfalse = 0;
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string vr = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    ValueRef *v = symbol_table_lookup(&parser->symbol_table, vr);
                    if (v) VecValueRef_push_back(parser->arena, &operands, v);
                    nfalse++;
                    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                } else if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    int angle=0;
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                        parser_next_token(parser);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
            size_t n = op->n_attributes; Attribute **attrs = op->attributes;
            Attribute *an = arena_alloc(parser->arena, Attribute);
            an->name = str_lit("_nfalse"); an->kind = ATTR_KIND_INTEGER; an->data.integer_value = nfalse;
            Attribute **na = arena_alloc_array(parser->arena, Attribute*, n+1);
            for (size_t i=0;i<n;i++) na[i]=attrs[i];
            na[n]=an;
            op->attributes=na; op->n_attributes=(int)(n+1);
        }
    }
    op->operands = operands.data; op->n_operands = operands.size;
    // Store private attributes for classic printing; append _true/_false without dropping existing counts
    size_t n0 = op->n_attributes; Attribute **attrs0 = op->attributes;
    Attribute **na0 = arena_alloc_array(parser->arena, Attribute*, n0 + 2);
    for (size_t i=0;i<n0;i++) na0[i]=attrs0[i];
    na0[n0] = arena_alloc(parser->arena, Attribute);
    na0[n0]->name = str_lit("_true");
    na0[n0]->kind = ATTR_KIND_STRING;
    na0[n0]->data.string_value = ttrue;
    na0[n0+1] = arena_alloc(parser->arena, Attribute);
    na0[n0+1]->name = str_lit("_false");
    na0[n0+1]->kind = ATTR_KIND_STRING;
    na0[n0+1]->data.string_value = tfalse;
    op->attributes = na0; op->n_attributes = (int)(n0 + 2);
    // Optional loc
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
}

void parse_linalg_fill(Parser *parser, Operation *op) {
    // Parse linalg.fill ins(%val : type) outs(%tensor : type)

    // Parse ins()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("ins"))) {
        parser_expect(parser, TK_NAME);  // "ins"
        parser_expect(parser, TK_LPAREN);
        // Consume operand with type annotation
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                parser_expect(parser, TK_REGISTER);
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    // Consume type
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        parser_next_token(parser);
                    }
                }
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    // Parse outs()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("outs"))) {
        parser_expect(parser, TK_NAME);  // "outs"
        parser_expect(parser, TK_LPAREN);
        // Consume operand with type annotation
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                parser_expect(parser, TK_REGISTER);
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    // Consume type
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        parser_next_token(parser);
                    }
                }
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    // linalg.fill has no result types
    op->n_result_types = 0;

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
}

void parse_affine_load(Parser *parser, Operation *op) {
    // Parse affine.load %memref[%index] : memref<type>
    VecValueRef operands; VecValueRef_reserve(parser->arena, &operands, 2);

    // Parse memref operand
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *memref_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!memref_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, memref_operand);
    }

    // Parse indices in brackets
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *idx = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!idx) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValueRef_push_back(parser->arena, &operands, idx);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Parse ": memref<elementtype>" and infer result type as elementtype
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        // For affine.load, the result type is the element type of the memref
        // Parse memref<elementtype> and extract elementtype
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("memref"))) {
            parser_expect(parser, TK_NAME); // "memref"
            if (parser_peek(parser, TK_LANGLE)) {
                parser_expect(parser, TK_LANGLE);
                // Parse type dimensions and element type
                string element_type = str_lit("unknown");

                if (parser_peek(parser, TK_TYPE_DIM)) {
                    string type_dim = parser_token_str(parser);
                    parser_expect(parser, TK_TYPE_DIM);

                    // Extract element type from "24xi16" -> "i16"
                    // Find the last 'x' and take everything after it
                    size_t last_x = 0;
                    for (size_t i = 0; i < type_dim.size; i++) {
                        if (type_dim.str[i] == 'x') {
                            last_x = i;
                        }
                    }
                    if (last_x > 0 && last_x + 1 < type_dim.size) {
                        element_type = str_substr(type_dim, last_x + 1, type_dim.size - last_x - 1);
                    }
                }

                // If there's a separate element type token after TK_TYPE_DIM
                if (parser_peek(parser, TK_NAME)) {
                    element_type = parser_token_str(parser);
                    parser_expect(parser, TK_NAME);
                }

                // Set result type to element type
                op->n_result_types = 1;
                op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                op->result_types[0] = arena_alloc(parser->arena, Type);
                op->result_types[0] = parse_type_from_string(parser->arena, element_type);

                parser_expect(parser, TK_RANGLE);
            }
        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
}

void parse_index_constant(Parser *parser, Operation *op) {
    // Parse index.constant <value>
    // This should produce a result of type 'index'

    // Consume the constant value
    if (parser_peek(parser, TK_INTEGER)) {
        parser_expect(parser, TK_INTEGER);
    } else {
        // Skip any tokens until we find what we expect
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
               !parser_peek(parser, TK_NAME) && !parser_peek(parser, TK_COLON)) {
            parser_next_token(parser);
        }
    }

    // index.constant produces a result of type 'index'
    op->n_result_types = 1;
    op->result_types = arena_alloc_array(parser->arena, Type*, 1);
    op->result_types[0] = arena_alloc(parser->arena, Type);
    op->result_types[0] = parse_type_from_string(parser->arena, str_lit("index"));


    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
}

void parse_tensor_splat(Parser *parser, Operation *op) {
    // Parse tensor.splat %value[%dim1, %dim2] : tensor<type>
    VecValueRef operands; VecValueRef_reserve(parser->arena, &operands, 3);

    // Parse value operand
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *value_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!value_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, value_operand);
    }

    // Parse dimensions in brackets
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *dim = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!dim) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValueRef_push_back(parser->arena, &operands, dim);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Parse ": tensor<type>" and use that as result type
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("tensor"))) {
            parser_expect(parser, TK_NAME); // "tensor"
            if (parser_peek(parser, TK_LANGLE)) {
                parser_expect(parser, TK_LANGLE);

                // Capture the entire tensor type
                string tensor_type = str_lit("tensor<");
                int depth = 1;
                while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                    string token_str = parser_token_str(parser);
                    if (parser_peek(parser, TK_LANGLE)) depth++;
                    else if (parser_peek(parser, TK_RANGLE)) depth--;

                    tensor_type = str_concat(parser->arena, tensor_type, token_str);
                    parser_next_token(parser);
                }

                // Set result type to the full tensor type
                op->n_result_types = 1;
                op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                op->result_types[0] = arena_alloc(parser->arena, Type);
                op->result_types[0] = parse_type_from_string(parser->arena, tensor_type);

            }
        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
}

void parse_arith_select(Parser *parser, Operation *op) {
    // Parse arith.select %cond, %true_val, %false_val : cond_type, val_type
    VecValueRef operands; VecValueRef_reserve(parser->arena, &operands, 3);

    // Parse three operands
    for (int i = 0; i < 3; i++) {
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValueRef_push_back(parser->arena, &operands, operand);
            if (i < 2 && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        } else {
            parser_error(parser, str_lit("Expected register operand"), parser->first, parser->last);
            return;
        }
    }

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Parse ": type1, type2" - result type is type2 (the value type)
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip first type (condition type)
        while (!parser_peek(parser, TK_COMMA) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE)) {
            parser_next_token(parser);
        }

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);

            // Parse the second type (result type)
            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) &&
                   !parser_peek(parser, TK_NEWLINE) &&
                   !(parser_peek(parser, TK_NAME) &&
                     str_eq(parser_token_str(parser), str_lit("loc")))) {
                if (result_type.size > 0) {
                    result_type = str_concat(parser->arena, result_type, parser_token_str(parser));
                } else {
                    result_type = parser_token_str(parser);
                }
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            // Set result type
            op->n_result_types = 1;
            op->result_types = arena_alloc_array(parser->arena, Type*, 1);
            op->result_types[0] = arena_alloc(parser->arena, Type);
            op->result_types[0] = parse_type_from_string(parser->arena, result_type);

        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
}

void parse_tt_call(Parser *parser, Operation *op) {
    // Parse tt.call @function(%args) : (arg_types) -> result_type

    // Parse function name (@name is tokenized as TK_FUNCTION_NAME)
    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture callee name including '@'
        string fname = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        // Store as attribute 'callee'
        size_t n = op->n_attributes; Attribute **attrs = op->attributes;
        if (!attrs) { attrs = arena_alloc_array(parser->arena, Attribute*, 1); n = 0; }
        else {
            Attribute **new_attrs = arena_alloc_array(parser->arena, Attribute*, n+1);
            for (size_t i=0;i<n;i++) new_attrs[i]=attrs[i]; attrs = new_attrs;
        }
        attrs[n] = arena_alloc(parser->arena, Attribute);
        attrs[n]->name = str_lit("callee"); attrs[n]->kind = ATTR_KIND_STRING; attrs[n]->data.string_value = fname;
        op->attributes = attrs; op->n_attributes = n+1;
    }

    // Parse arguments in parentheses
    VecValueRef operands; VecValueRef_reserve(parser->arena, &operands, 4);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValueRef_push_back(parser->arena, &operands, operand);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Parse ": (arg_types) -> result_type"
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip argument types in parentheses
        if (parser_peek(parser, TK_LPAREN)) {
            int depth = 0;
            do {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            } while (depth > 0 && !parser_peek(parser, TK_EOF));
        }

        // Parse "-> result_type"
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);

            // Parse result type
            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
                   (!parser_peek(parser, TK_NAME) || !str_eq(parser_token_str(parser), str_lit("loc")))) {
                if (result_type.size > 0) {
                    result_type = str_concat(parser->arena, result_type, parser_token_str(parser));
                } else {
                    result_type = parser_token_str(parser);
                }
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            // Set result type
            op->n_result_types = 1;
            op->result_types = arena_alloc_array(parser->arena, Type*, 1);
            op->result_types[0] = arena_alloc(parser->arena, Type);
            op->result_types[0] = parse_type_from_string(parser->arena, result_type);

        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
}

void parse_tensor_collapse_shape(Parser *parser, Operation *op) {
    // Parse tensor.collapse_shape %tensor [[indices]] : input_type into result_type
    VecValueRef operands; VecValueRef_reserve(parser->arena, &operands, 1);

    // Parse tensor operand
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *tensor_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!tensor_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, tensor_operand);
    }

    // Parse [[indices]] - skip the bracket notation
    if (parser_peek(parser, TK_LBRACKET)) {
        int depth = 0;
        do {
            if (parser_peek(parser, TK_LBRACKET)) depth++;
            else if (parser_peek(parser, TK_RBRACKET)) depth--;
            parser_next_token(parser);
        } while (depth > 0 && !parser_peek(parser, TK_EOF));
    }

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Parse ": input_type into result_type"
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip input type until "into"
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
               (!parser_peek(parser, TK_NAME) || !str_eq(parser_token_str(parser), str_lit("into")))) {
            parser_next_token(parser);
        }

        // Parse "into result_type"
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("into"))) {
            parser_expect(parser, TK_NAME); // "into"

            // Parse result type
            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
                   (!parser_peek(parser, TK_NAME) || !str_eq(parser_token_str(parser), str_lit("loc")))) {
                if (result_type.size > 0) {
                    result_type = str_concat(parser->arena, result_type, parser_token_str(parser));
                } else {
                    result_type = parser_token_str(parser);
                }
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            // Set result type
            op->n_result_types = 1;
            op->result_types = arena_alloc_array(parser->arena, Type*, 1);
            op->result_types[0] = arena_alloc(parser->arena, Type);
            op->result_types[0] = parse_type_from_string(parser->arena, result_type);

        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
}

void parse_generic_operation(Parser *parser, Operation *op) {
    // Skip non-structural tokens (e.g., cmp predicates) until operands/attrs/types
    while (!parser_peek(parser, TK_EOF) &&
           !(parser_peek(parser, TK_REGISTER) || parser_peek(parser, TK_LBRACKET) ||
             parser_peek(parser, TK_COLON) || parser_peek(parser, TK_LBRACE) ||
             parser_peek(parser, TK_LBRACE_END) || parser_peek(parser, TK_NEWLINE) ||
             parser_peek(parser, TK_LPAREN_BRACE))) {
        parser_next_token(parser);
    }

    // Parse operands until attributes, result type, or region begin
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 4);

    while ((parser_peek(parser, TK_REGISTER) || parser_peek(parser, TK_LBRACKET)) &&
           !parser_peek(parser, TK_COLON) && !parser_peek(parser, TK_LBRACE)) {
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValueRef_push_back(parser->arena, &operands, operand);
        } else if (parser_peek(parser, TK_LBRACKET)) {
            // Skip indexing syntax like [%arg1]
            parser_expect(parser, TK_LBRACKET);
            while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str2 = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    ValueRef *operand2 = symbol_table_lookup(&parser->symbol_table, reg_str2);
                    if (!operand2) {
                        parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                        return;
                    }
                    VecValueRef_push_back(parser->arena, &operands, operand2);
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RBRACKET);
        }

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
        } else {
            break;
        }
    }

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Attributes and result types are parsed generically later
    // Handle generic attributes and result types before scanning for regions
    parse_generic_attrs_and_result_type(parser, op);
    // Capture trailing loc() immediately if present to avoid getting skipped
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }

    // Parse regions (if any), for now we assume 0 or 1 regions
    while (!(parser_peek(parser, TK_LBRACE_END)
             || parser_peek(parser, TK_NEWLINE)
             || parser_peek(parser, TK_LPAREN_BRACE))) {
        parser_next_token(parser);
    }
    bool lparen_brace = false;
    if (parser_peek(parser, TK_LPAREN_BRACE)) {
        lparen_brace = true;
        parser_next_token(parser);
    }
    if (parser_peek(parser, TK_LBRACE_END)) {
        // Before consuming a region, parse any trailing attributes/result types
        parse_generic_attrs_and_result_type(parser, op);
        Region *region = parse_region(parser);

        Region **regions = arena_alloc(parser->arena, Region*);
        regions[0] = region;
        op->regions = regions;
        op->n_regions = 1;
        // After region, there can be another signature/attrs like ": (tys) -> ty"
        parse_generic_attrs_and_result_type(parser, op);
    }
    if (lparen_brace) {
        parser_expect(parser, TK_RPAREN);
        while (!parser_peek(parser, TK_NEWLINE)) {
            parser_next_token(parser);
        }
    }
    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }
}

void parse_tt_func(Parser *parser, Operation *op) {
    // Parse tt.func public @function_name(%arg0: type, %arg1: type, ...)

    // Capture visibility keyword if present
    string visibility = str_lit("private");  // default
    if (parser_peek(parser, TK_NAME) && (str_eq(parser_token_str(parser), str_lit("public")) || str_eq(parser_token_str(parser), str_lit("private")))) {
        visibility = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
    }
    
    // Store visibility as attribute
    op->n_attributes = 1;
    op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
    op->attributes[0] = arena_alloc(parser->arena, Attribute);
    op->attributes[0]->kind = ATTR_KIND_STRING;
    op->attributes[0]->data.string_value = visibility;
    op->attributes[0]->name = str_lit("visibility");

    // Parse @function_name
    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture function symbol name into an attribute for printing
        string fname_with_at = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        
        // Remove the '@' prefix to get just the function name
        string fname = str_substr(fname_with_at, 1, fname_with_at.size - 1);
        
        // Store as string attribute 'sym_name'
        size_t n = op->n_attributes;
        Attribute **new_attrs = arena_alloc_array(parser->arena, Attribute*, n+1);
        for (size_t i = 0; i < n; i++) new_attrs[i] = op->attributes[i];
        new_attrs[n] = arena_alloc(parser->arena, Attribute);
        new_attrs[n]->kind = ATTR_KIND_STRING;
        new_attrs[n]->data.string_value = fname;
        new_attrs[n]->name = str_lit("sym_name");
        op->attributes = new_attrs;
        op->n_attributes = n+1;
    } else {
        // Try to skip tokens until we find a parenthesis
        while (!parser_peek(parser, TK_LPAREN) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
            parser_next_token(parser);
        }
    }

    // Parse function arguments
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);

        // Use a vector to collect arguments as we parse them
        VecValueRef func_args;
        VecValueRef_reserve(parser->arena, &func_args, 8);

        // Parse arguments directly in one pass
        int paren_depth = 0;
        while (!parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LPAREN)) {
                paren_depth++;
                parser_next_token(parser);
            } else if (parser_peek(parser, TK_RPAREN)) {
                if (paren_depth == 0) {
                    // Found the main closing paren
                    break;
                }
                paren_depth--;
                parser_next_token(parser);
            } else if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);

                // Parse colon and type
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);

                    ValueRef *arg = create_value_ref(parser->arena, BLOCK_ARG);
                    arg->register_name = reg_str;
                    arg->result_index = func_args.size;
                    arg->type = arena_alloc(parser->arena, Type);

                    // Parse the argument type
                    if (parser_peek(parser, TK_EXCLAMATION)) {
                        parser_expect(parser, TK_EXCLAMATION);

                        if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                            string type_name = parser_token_str(parser);
                            if (parser_peek(parser, TK_NAME)) {
                                parser_expect(parser, TK_NAME);
                            } else {
                                parser_expect(parser, TK_NAME_DOT_NAME);
                            }

                            // Handle complex types like !tt.ptr<f32, 1>
                            if (str_eq(type_name, str_lit("tt.ptr"))) {
                                if (parser_peek(parser, TK_LANGLE)) {
                                    parser_expect(parser, TK_LANGLE);

                                    // Parse everything inside angle brackets
                                    string type_content = str_lit("");
                                    while (!parser_peek(parser, TK_RANGLE) && !parser_peek(parser, TK_EOF)) {
                                        string token_str = parser_token_str(parser);
                                        type_content = str_concat(parser->arena, type_content, token_str);
                                        parser_next_token(parser);
                                    }

                                    if (parser_peek(parser, TK_RANGLE)) {
                                        parser_expect(parser, TK_RANGLE);
                                        arg->type = parse_type_from_string(parser->arena, str_concat(parser->arena, str_lit("!tt.ptr<"), str_concat(parser->arena, type_content, str_lit(">"))));
                                    }
                                } else {
                                    arg->type = parse_type_from_string(parser->arena, str_lit("!tt.ptr"));
                                }
                            } else {
                                arg->type = parse_type_from_string(parser->arena, str_concat(parser->arena, str_lit("!"), type_name));
                            }
                        } else {
                            arg->type = parse_type_from_string(parser->arena, str_lit("!unknown"));
                        }
                    } else if (parser_peek(parser, TK_NAME)) {
                        // Simple type like i32
                        string type_name = parser_token_str(parser);
                        parser_expect(parser, TK_NAME);
                        arg->type = parse_type_from_string(parser->arena, type_name);
                    } else {
                        arg->type = parse_type_from_string(parser->arena, str_lit("unknown"));
                    }

                    // Optional per-arg attribute dict: { ... }
                    if (parser_peek(parser, TK_LBRACE)) {
                        int depth = 0;
                        // Try to detect simple 'tt.divisibility = <int> : <type>' pattern
                        // without constructing a full attribute AST.
                        // Consume '{'
                        parser_expect(parser, TK_LBRACE);
                        depth++;
                        // Read tokens until matching '}'
                        while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                            if (parser_peek(parser, TK_LBRACE)) { depth++; parser_next_token(parser); continue; }
                            if (parser_peek(parser, TK_RBRACE)) { depth--; parser_next_token(parser); if (depth==0) break; else continue; }
                            // Match name possibly with dots
                            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                                string name = parser_token_str(parser);
                                parser_next_token(parser);
                                if (str_eq(name, str_lit("tt.divisibility"))) {
                                    // Expect '=' integer ':' type
                                    if (parser_peek(parser, TK_EQUAL)) parser_expect(parser, TK_EQUAL);
                                    if (parser_peek(parser, TK_INTEGER)) {
                                        string ival = parser_token_str(parser);
                                        // parse integer
                                        int64_t v = 0; for (size_t k=0;k<ival.size;k++){ char c=ival.str[k]; if (c>='0' && c<='9') v = v*10 + (c-'0'); }
                                        parser_expect(parser, TK_INTEGER);
                                        // optional ':' type
                                        Type *dtype = NULL;
                                        if (parser_peek(parser, TK_COLON)) {
                                            parser_expect(parser, TK_COLON);
                                            string tstr = str_lit("");
                                            // Reuse type parsing helper from mlir_parser.c via forward decl
                                            // Here, mimic parse_type_string: collect until '}' or ',' or 'loc'
                                            int angle = 0;
                                            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_COMMA)) {
                                                if (parser_peek(parser, TK_LANGLE)) angle++;
                                                else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                                                if (angle==0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                                                string tok = parser_token_str(parser);
                                                tstr = tstr.size ? str_concat(parser->arena, tstr, tok) : tok;
                                                parser_next_token(parser);
                                                if (angle==0 && (parser_peek(parser, TK_RBRACE) || parser_peek(parser, TK_COMMA))) break;
                                            }
                                            dtype = parse_type_from_string(parser->arena, tstr);
                                        }
                                        arg->has_divisibility = true;
                                        arg->divisibility_value = v;
                                        arg->divisibility_type = dtype ? dtype : parse_type_from_string(parser->arena, str_lit("i32"));
                                    }
                                } else if (str_eq(name, str_lit("tt.max_divisibility"))) {
                                    // Expect '=' integer ':' type
                                    if (parser_peek(parser, TK_EQUAL)) parser_expect(parser, TK_EQUAL);
                                    if (parser_peek(parser, TK_INTEGER)) {
                                        string ival = parser_token_str(parser);
                                        // parse integer
                                        int64_t v = 0; for (size_t k=0;k<ival.size;k++){ char c=ival.str[k]; if (c>='0' && c<='9') v = v*10 + (c-'0'); }
                                        parser_expect(parser, TK_INTEGER);
                                        // optional ':' type
                                        Type *dtype = NULL;
                                        if (parser_peek(parser, TK_COLON)) {
                                            parser_expect(parser, TK_COLON);
                                            string tstr = str_lit("");
                                            int angle = 0;
                                            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_COMMA)) {
                                                if (parser_peek(parser, TK_LANGLE)) angle++;
                                                else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                                                if (angle==0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                                                string tok = parser_token_str(parser);
                                                tstr = tstr.size ? str_concat(parser->arena, tstr, tok) : tok;
                                                parser_next_token(parser);
                                                if (angle==0 && (parser_peek(parser, TK_RBRACE) || parser_peek(parser, TK_COMMA))) break;
                                            }
                                            dtype = parse_type_from_string(parser->arena, tstr);
                                        }
                                        arg->has_max_divisibility = true;
                                        arg->max_divisibility_value = v;
                                        arg->max_divisibility_type = dtype ? dtype : parse_type_from_string(parser->arena, str_lit("i32"));
                                    }
                                }
                            } else {
                                parser_next_token(parser);
                            }
                        }
                    }

                    // Optional trailing per-arg loc()
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                        arg->location = parse_loc(parser);
                    }

                    VecValueRef_push_back(parser->arena, &func_args, arg);
                }
            } else {
                parser_next_token(parser);
            }
        }

        // Convert vector to array
        op->n_operands = func_args.size;
        op->operands = arena_alloc_array(parser->arena, ValueRef*, func_args.size);
        for (size_t i = 0; i < func_args.size; i++) {
            op->operands[i] = func_args.data[i];
        }

        if (parser_peek(parser, TK_RPAREN)) {
            parser_expect(parser, TK_RPAREN);
        } else {
        }
    }

    // Optionally capture function attributes: attributes { ... }
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("attributes"))) {
        parser_expect(parser, TK_NAME); // attributes
        if (parser_peek(parser, TK_LBRACE)) {
            int depth = 0; parser_expect(parser, TK_LBRACE); depth++;
            // Conservatively skip content; generic op attribute parser can be used later if needed
            while (depth>0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LBRACE)) { depth++; }
                else if (parser_peek(parser, TK_RBRACE)) { depth--; }
                parser_next_token(parser);
            }
        }
    }

    // Skip any remaining tokens until function body
    while (!parser_peek(parser, TK_LBRACE_END)) {
        parser_next_token(parser);
    }

    // Parse the function body region with special handling for function arguments
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for this region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Register function arguments in the new scope
    for (int i = 0; i < op->n_operands; i++) {
        if (op->operands[i]) {
            symbol_table_add_value(parser->arena, &parser->symbol_table, op->operands[i]->register_name, op->operands[i]);
        }
    }

    // Parse blocks
    VecBlock blocks;
    VecBlock_reserve(parser->arena, &blocks, 8);
    while (!parser_peek(parser, TK_RBRACE)) {
        Block *block = parse_block(parser);
        VecBlock_push_back(parser->arena, &blocks, block);
    }
    parser_expect(parser, TK_RBRACE);

    // Pop scope when leaving region
    symbol_table_pop_scope(&parser->symbol_table);

    // Create region
    Region *region = arena_alloc(parser->arena, Region);
    region->blocks = blocks.data;
    region->n_blocks = blocks.size;

    // Set up block arguments
    // Note: For tt.func, we don't copy arguments to the first block to avoid duplication
    // since the arguments are already printed in the function signature
    if (region->n_blocks > 0 && op->n_operands > 0) {
        Block *entry_block = region->blocks[0];
        // Don't set block arguments for tt.func - they're already in the function signature
        entry_block->n_arguments = 0;
        entry_block->arguments = NULL;
    }

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }

    Region **regions = arena_alloc(parser->arena, Region*);
    regions[0] = region;
    op->regions = regions;
    op->n_regions = 1;
}

void parse_func_func(Parser *parser, Operation *op) {
    // Capture optional visibility and function name
    string visibility = str_lit("");
    string fname = str_lit("");
    while (!parser_peek(parser, TK_LPAREN) && !parser_peek(parser, TK_EOF)) {
        if (parser_peek(parser, TK_NAME)) {
            string nm = parser_token_str(parser);
            if (str_eq(nm, str_lit("private")) || str_eq(nm, str_lit("public"))) {
                visibility = nm; parser_expect(parser, TK_NAME); continue;
            }
        }
        if (parser_peek(parser, TK_FUNCTION_NAME)) {
            fname = parser_token_str(parser); parser_expect(parser, TK_FUNCTION_NAME); continue;
        }
        parser_next_token(parser);
    }
    // Parse argument list
    VecValueRef args; VecValueRef_reserve(parser->arena, &args, 8);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        bool first = true;
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                // Optional type annotation
                Type *ty = NULL;
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    string t = str_lit("");
                    if (parse_type_string(parser, &t)) {
                        ty = parse_type_from_string(parser->arena, t);

                    }
                }
                ValueRef *arg = create_value_ref(parser->arena, BLOCK_ARG);
                arg->register_name = reg;
                arg->type = ty;
                VecValueRef_push_back(parser->arena, &args, arg);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }
    // Optional return type sequence after '->' (capture text conservatively)
    string ret_sig = str_lit("");
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        // Capture until '{' or loc
        while (!(parser_peek(parser, TK_LBRACE_END) || parser_peek(parser, TK_EOF))) {
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            ret_sig = str_concat(parser->arena, ret_sig, parser_token_str(parser));
            parser_next_token(parser);
        }
    }
    // Parse body region with caret block labels and arguments
    Region *region = parse_region(parser);
    op->regions = arena_alloc_array(parser->arena, Region*, 1);
    op->regions[0] = region;
    op->n_regions = 1;

    // Store attributes for classic printing
    size_t n = op->n_attributes; Attribute **attrs = op->attributes;
    size_t add = 0; if (visibility.size>0) add++; if (fname.size>0) add++; if (ret_sig.size>0) add++;
    if (add > 0) {
        Attribute **new_attrs = arena_alloc_array(parser->arena, Attribute*, n+add);
        for (size_t i=0;i<n;i++) new_attrs[i]=attrs[i]; size_t idx=n;
        if (visibility.size>0){ new_attrs[idx]=arena_alloc(parser->arena, Attribute); new_attrs[idx]->name=str_lit("visibility"); new_attrs[idx]->kind=ATTR_KIND_STRING; new_attrs[idx]->data.string_value=visibility; idx++; }
        if (fname.size>0){ new_attrs[idx]=arena_alloc(parser->arena, Attribute); new_attrs[idx]->name=str_lit("sym_name"); new_attrs[idx]->kind=ATTR_KIND_STRING; new_attrs[idx]->data.string_value=fname; idx++; }
        if (ret_sig.size>0){ new_attrs[idx]=arena_alloc(parser->arena, Attribute); new_attrs[idx]->name=str_lit("ret"); new_attrs[idx]->kind=ATTR_KIND_STRING; new_attrs[idx]->data.string_value=ret_sig; idx++; }
        op->attributes = new_attrs; op->n_attributes = idx;
    }
}

void parse_scf_if(Parser *parser, Operation *op) {
    // Parse condition operand: scf.if %condition -> ...
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 1);
    
    if (parser_peek(parser, TK_REGISTER)) {
        string cond_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *condition = symbol_table_lookup(&parser->symbol_table, cond_str);
        if (condition) {
            VecValueRef_push_back(parser->arena, &operands, condition);
        }
    }
    
    // Set operands
    op->operands = operands.data;
    op->n_operands = operands.size;
    
    // Capture optional result types: '-> (type, ... )'
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);
                if (parser_peek(parser, TK_LPAREN)) {
                    parser_expect(parser, TK_LPAREN);
                    // Parse a single result type conservatively
                    string t = str_lit("");
                    if (parse_type_string(parser, &t)) {
                        op->n_result_types = 1;
                        op->result_types = arena_alloc_array(parser->arena, Type*, 1);
                        op->result_types[0] = arena_alloc(parser->arena, Type);
                        op->result_types[0] = parse_type_from_string(parser->arena, t);
                    }
                    // Consume rest until ')'
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
                    if (parser_peek(parser, TK_RPAREN)) parser_expect(parser, TK_RPAREN);
                }
        } else {
            parser_next_token(parser);
        }
    }
    int n_regions = 1;
    Region *region1 = parse_region(parser);
    Region *region2 = NULL;

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("else"))) {
            parser_expect(parser, TK_NAME);
            region2 = parse_region(parser);
            n_regions++;
        }
    }

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }

    Region **regions = arena_alloc_array(parser->arena, Region*, n_regions);
    regions[0] = region1;
    if (region2) {
        regions[1] = region2;
    }
    op->regions = regions;
    op->n_regions = n_regions;
}

void parse_scf_for(Parser *parser, Operation *op) {
    // Parse scf.for %iv = %lb to %ub step %step
    //        [iter_args(%arg = %init, ...)] [-> (types...)]  : iv_type { ... }

    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 4);

    // Parse loop variable: %iv
    ValueRef *loop_var = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string loop_var_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        loop_var = create_value_ref(parser->arena, BLOCK_ARG);
        loop_var->register_name = loop_var_name;
        // Type assigned from trailing ": <type>" if present; default later
        loop_var->type = NULL;


        // Expect =
        parser_expect(parser, TK_EQUAL);

        // Parse %start operand
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            ValueRef *start_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!start_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValueRef_push_back(parser->arena, &operands, start_operand);
        }

        // Expect "to"
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
            parser_expect(parser, TK_NAME);
        }

        // Parse %end operand
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            ValueRef *end_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!end_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValueRef_push_back(parser->arena, &operands, end_operand);
        }

        // Expect "step"
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) {
            parser_expect(parser, TK_NAME);
        }

        // Parse %step operand
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            ValueRef *step_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!step_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValueRef_push_back(parser->arena, &operands, step_operand);
        }
    }

    // Parse iter_args(%iter_var1 = %init1, %iter_var2 = %init2, ...)
    VecValueRef iter_vars;
    VecValueRef_reserve(parser->arena, &iter_vars, 4);

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("iter_args"))) {
        parser_expect(parser, TK_NAME); // consume "iter_args"

        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string iter_var_name = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);

                    ValueRef *iter_var = create_value_ref(parser->arena, BLOCK_ARG);
                    // Keep the original SSA name for printing and symbol mapping
                    iter_var->register_name = iter_var_name;
                    iter_var->type = NULL; // Determined from init operand


                    VecValueRef_push_back(parser->arena, &iter_vars, iter_var);

                    // Expect =
                    parser_expect(parser, TK_EQUAL);

                    // Parse %init operand
                    if (parser_peek(parser, TK_REGISTER)) {
                        string reg_str = parser_token_str(parser);
                        parser_expect(parser, TK_REGISTER);
                        consume_optional_hash_selector(parser);
                        consume_optional_hash_selector(parser);

                        ValueRef *init_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                        if (!init_operand) {
                            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                            return;
                        }
                        VecValueRef_push_back(parser->arena, &operands, init_operand);
                    }

                    // Handle comma for multiple iter_args
                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    } else {
                        break;
                    }
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Set operands for the scf.for operation
    op->operands = operands.data;
    op->n_operands = operands.size;
    
    // Parse optional result types list '-> (type, ...)' before region
    Type **iter_result_types = NULL;
    size_t n_iter_results = 0;
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            // Parse one or more types separated by commas
            // Minimal manual growth: collect types with manual growth
            size_t cap = 4; size_t sz = 0;
            Type **tmp = arena_alloc_array(parser->arena, Type*, cap);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                string t = str_lit("");
                if (!parse_type_string(parser, &t)) break;
                if (sz >= cap) {
                    size_t ncap = cap * 2;
                    Type **nt = arena_alloc_array(parser->arena, Type*, ncap);
                    for (size_t i = 0; i < sz; i++) nt[i] = tmp[i];
                    tmp = nt; cap = ncap;
                }
                tmp[sz++] = parse_type_from_string(parser->arena, t);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RPAREN);
            iter_result_types = tmp;
            n_iter_results = sz;
        }
    }

    // Optional trailing ": <iv_type>" before region
    Type *iv_type = NULL;
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        string t = str_lit("");
        if (parse_type_string(parser, &t)) {
            iv_type = parse_type_from_string(parser->arena, t);
        }
    }

    // Parse the region, but first register loop arguments in the new scope
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for this region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Create a single block with loop variable and iter_args as block arguments
    VecBlock blocks;
    VecBlock_reserve(parser->arena, &blocks, 1);

    // Create the block
    Block *block = arena_alloc(parser->arena, Block);

    // Create block arguments for loop variable and iter_args
    VecValueRef block_args;
    VecValueRef_reserve(parser->arena, &block_args, 2);

    if (loop_var) {
        // Create a new ValueRef for the block argument using the original loop variable name
        ValueRef *loop_block_arg = create_value_ref(parser->arena, BLOCK_ARG);
        loop_block_arg->register_name = loop_var->register_name;
        // Use parsed iv type if available; otherwise default to the lb type or i32
        if (iv_type) {
            loop_block_arg->type = iv_type;
        } else if (operands.size > 0 && operands.data[0] && operands.data[0]->type) {
            loop_block_arg->type = operands.data[0]->type;
        } else {
            loop_block_arg->type = parse_type_from_string(parser->arena, str_lit("i32"));
        }

        loop_block_arg->result_index = 0;
        loop_block_arg->def = block;

        VecValueRef_push_back(parser->arena, &block_args, loop_block_arg);

        // Register in symbol table using original loop variable name
        symbol_table_add_value(parser->arena, &parser->symbol_table, loop_var->register_name, loop_block_arg);
    }

    // Create block arguments for all iter_args
    for (size_t i = 0; i < iter_vars.size; i++) {
        ValueRef *iter_var = iter_vars.data[i];
        // Create a new ValueRef for the block argument using original name
        ValueRef *iter_block_arg = create_value_ref(parser->arena, BLOCK_ARG);
        iter_block_arg->register_name = iter_var->register_name;
        // Type of iter arg is the type of its init operand (operands[3+i])
        if (op->n_operands >= 4 + (int)i && operands.data[3 + i] && operands.data[3 + i]->type) {
            iter_block_arg->type = operands.data[3 + i]->type;
        } else {
            iter_block_arg->type = parse_type_from_string(parser->arena, str_lit("unknown"));
        }

        iter_block_arg->result_index = i + 1;
        iter_block_arg->def = block;

        VecValueRef_push_back(parser->arena, &block_args, iter_block_arg);

        // Register in symbol table using original iter_args variable name
        symbol_table_add_value(parser->arena, &parser->symbol_table, iter_var->register_name, iter_block_arg);
    }

    block->arguments = block_args.data;
    block->n_arguments = block_args.size;

    // Parse operations inside the block
    VecOperation operations;
    VecOperation_reserve(parser->arena, &operations, 16);
    while (!parser_peek(parser, TK_RBRACE)) {
        Operation *op = parse_operation(parser);
        VecOperation_push_back(parser->arena, &operations, op);
        parser_expect(parser, TK_NEWLINE);

        // Skip empty lines
        while (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        }
    }

    block->operations = operations.data;
    block->n_operations = operations.size;

    VecBlock_push_back(parser->arena, &blocks, block);

    parser_expect(parser, TK_RBRACE);

    // Pop scope when exiting region
    symbol_table_pop_scope(&parser->symbol_table);

    // Create the region containing the block
    Region *region = arena_alloc(parser->arena, Region);
    region->blocks = blocks.data;
    region->n_blocks = blocks.size;

    // Assign region to operation
    op->regions = arena_alloc_array(parser->arena, Region*, 1);
    op->regions[0] = region;
    op->n_regions = 1;

    // Assign parsed iter result types to operation
    if (n_iter_results > 0) {
        op->n_result_types = (int)n_iter_results;
        op->result_types = arena_alloc_array(parser->arena, Type*, n_iter_results);
        for (size_t i = 0; i < n_iter_results; i++) op->result_types[i] = iter_result_types[i];
    }

    // Handle optional location attribute after }
}

void parse_scf_while(Parser *parser, Operation *op) {
    while (!parser_peek(parser, TK_LBRACE_END)) {
        parser_next_token(parser);
    }
    int n_regions = 1;
    Region *region1 = parse_region(parser);
    Region *region2 = NULL;

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("do"))) {
            parser_expect(parser, TK_NAME);
            region2 = parse_region(parser);
            n_regions++;
        }
    }

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }

    Region **regions = arena_alloc_array(parser->arena, Region*, n_regions);
    regions[0] = region1;
    if (region2) {
        regions[1] = region2;
    }
    op->regions = regions;
    op->n_regions = n_regions;
}

void parse_scf_yield(Parser *parser, Operation *op) {
    // Parse scf.yield operands
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 2);

    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, operand);

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
        } else {
            break;
        }
    }

    op->operands = operands.data;
    op->n_operands = operands.size;
    op->op_type = OP_TYPE_SCF_YIELD;

    // Optionally parse ": types" then a trailing loc()
    if (parser_peek(parser, TK_COLON)) {
        // consume to end of types list
        do {
            parser_next_token(parser);
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
        } while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF));
    }
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
    // Skip to eol
    while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
}

void parse_return_operation(Parser *parser, Operation *op) {
    // Parse optional operands
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 2);
    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValueRef_push_back(parser->arena, &operands, operand);
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA); else break;
    }
    // Keep operands for classic and better generic fidelity
    op->operands = operands.data;
    op->n_operands = operands.size;

    // Consume any trailing ": ..." types or loc(), without assigning result types
    if (parser_peek(parser, TK_COLON)) {
        // Consume tokens until newline/brace
        do {
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                op->location = parse_loc(parser);
                break;
            }
            parser_next_token(parser);
        } while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF));
    }
    // Or a trailing loc() without preceding ':'
    if (!op->location && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }

    // Done with this op line
    while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
}

void parse_affine_for(Parser *parser, Operation *op) {
    // Expect induction variable
    ValueRef *ind_var = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string iv_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        // Create a placeholder for the block argument; registered later
        ind_var = create_value_ref(parser->arena, BLOCK_ARG);
        ind_var->register_name = iv_name;
        ind_var->type = arena_alloc(parser->arena, Type);
        ind_var->type = parse_type_from_string(parser->arena, str_lit("index"));


        // '=' lower bound
        if (parser_peek(parser, TK_EQUAL)) {
            parser_expect(parser, TK_EQUAL);
            // Consume a simple affine lower bound: integer or register/expression until 'to'
            while (!parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) break;
                if (parser_peek(parser, TK_LBRACE_END)) break;
                parser_next_token(parser);
            }
        }

        // 'to' upper bound
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
            parser_expect(parser, TK_NAME);
            // Consume simple upper bound until optional 'step' or '{'
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) break;
                parser_next_token(parser);
            }
        }

        // Optional 'step' expression
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) {
            parser_expect(parser, TK_NAME);
            // Consume step expression until '{'
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
                parser_next_token(parser);
            }
        }
    }

    // Now parse the loop region
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for induction variable
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Create single block and register induction variable as block argument
    Block *block = arena_alloc(parser->arena, Block);
    VecValueRef block_args; VecValueRef_reserve(parser->arena, &block_args, 1);
    if (ind_var) {
        ValueRef *iv_block_arg = create_value_ref(parser->arena, BLOCK_ARG);
        iv_block_arg->register_name = str_lit("%arg0");
        iv_block_arg->type = arena_alloc(parser->arena, Type);
        iv_block_arg->type = parse_type_from_string(parser->arena, str_lit("index"));
        iv_block_arg->result_index = 0;
        iv_block_arg->def = block;
        VecValueRef_push_back(parser->arena, &block_args, iv_block_arg);
        // Map original name (e.g., %i) to the block arg
        symbol_table_add_value(parser->arena, &parser->symbol_table, ind_var->register_name, iv_block_arg);
    }
    block->arguments = block_args.data;
    block->n_arguments = block_args.size;

    // Parse body operations
    bool prev_flag = parser->capture_trailing_comments;
    parser->capture_trailing_comments = true;
    VecOperation operations; VecOperation_reserve(parser->arena, &operations, 16);
    while (!parser_peek(parser, TK_RBRACE)) {
        Operation *inner = parse_operation(parser);
        // Capture trailing inline comment before consuming the newline
        if (parser_peek(parser, TK_NEWLINE)) {
            int64_t nl_pos = (int64_t)parser->first;
            if (nl_pos > 0) {
                int64_t line_end = nl_pos - 1;
                int64_t line_start = line_end;
                while (line_start > 0) {
                    unsigned char c = parser->input[line_start - 1];
                    if (c == '\n' || c == '\r') break;
                    line_start--;
                }
                int64_t comment_pos = -1;
                for (int64_t i = line_end - 1; i >= line_start; i--) {
                    if (parser->input[i] == '/' && parser->input[i + 1] == '/') { comment_pos = i; break; }
                }
                if (comment_pos >= 0) {
                    int64_t begin = comment_pos;
                    while (begin > line_start && parser->input[begin - 1] == ' ') begin--;
                    int64_t len = line_end - begin + 1;
                    if (len > 0) {
                        inner->trailing_comment = str_from_cstr_len_view((char*)parser->input + begin, len);
                    }
                }
            }
        } else {
            // Fallback: scan from end of op tokens to the end of this line
            string text = str_from_cstr_view((char*)parser->input);
            int64_t start = (int64_t)parser->last + 1;
            if (start < (int64_t)text.size) {
                int64_t end = (int64_t)text.size - 1;
                for (int64_t i = start; i < (int64_t)text.size; i++) {
                    char ch = text.str[i];
                    if (ch == '\n' || ch == '\r') { end = i - 1; break; }
                }
                if (end >= start) {
                    int64_t cpos = -1;
                    for (int64_t i = end - 1; i >= start; i--) {
                        if (text.str[i] == '/' && text.str[i+1] == '/') { cpos = i; break; }
                    }
                    if (cpos >= 0) {
                        int64_t begin = cpos;
                        while (begin > start && text.str[begin - 1] == ' ') begin--;
                        int64_t len = end - begin + 1;
                        if (len > 0) inner->trailing_comment = str_from_cstr_len_view(text.str + begin, len);
                    }
                }
            }
        }
        VecOperation_push_back(parser->arena, &operations, inner);
        parser_expect(parser, TK_NEWLINE);
        while (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
    }
    parser_expect(parser, TK_RBRACE);
    parser->capture_trailing_comments = prev_flag;

    // Optional trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }

    symbol_table_pop_scope(&parser->symbol_table);

    block->operations = operations.data;
    block->n_operations = operations.size;
    Region *region = arena_alloc(parser->arena, Region);
    region->blocks = arena_alloc_array(parser->arena, Block*, 1);
    region->blocks[0] = block;
    region->n_blocks = 1;
    op->regions = arena_alloc_array(parser->arena, Region*, 1);
    op->regions[0] = region;
    op->n_regions = 1;
}

void parse_gpu_launch(Parser *parser, Operation *op) {
    // Parse gpu.launch blocks(%arg2, %arg3, %arg4) in (...) threads(%arg5, %arg6, %arg7) in (...) {

    VecValueRef launch_args;
    VecValueRef_reserve(parser->arena, &launch_args, 16);

    // Parse blocks(...) section
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("blocks"))) {
        parser_expect(parser, TK_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            // Parse block arguments
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);

                    ValueRef *arg = create_value_ref(parser->arena, BLOCK_ARG);
                    arg->register_name = reg_str;
                    arg->result_index = launch_args.size;
                    arg->type = arena_alloc(parser->arena, Type);
                    arg->type = parse_type_from_string(parser->arena, str_lit("index"));

                    VecValueRef_push_back(parser->arena, &launch_args, arg);

                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Skip until threads(...) section
    while (!parser_peek(parser, TK_EOF) && !(parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("threads")))) {
        parser_next_token(parser);
    }

    // Parse threads(...) section
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("threads"))) {
        parser_expect(parser, TK_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            // Parse thread arguments
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);

                    ValueRef *arg = create_value_ref(parser->arena, BLOCK_ARG);
                    arg->register_name = reg_str;
                    arg->result_index = launch_args.size;
                    arg->type = arena_alloc(parser->arena, Type);
                    arg->type = parse_type_from_string(parser->arena, str_lit("index"));

                    VecValueRef_push_back(parser->arena, &launch_args, arg);

                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Skip until opening brace
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }

    // Parse the GPU launch region
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for GPU launch region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Register all GPU launch arguments in the new scope
    for (size_t i = 0; i < launch_args.size; i++) {
        symbol_table_add_value(parser->arena, &parser->symbol_table, launch_args.data[i]->register_name, launch_args.data[i]);
    }

    // Parse GPU launch body (single block)
    VecOperation operations;
    VecOperation_reserve(parser->arena, &operations, 16);
    while (!parser_peek(parser, TK_RBRACE)) {
        Operation *inner_op = parse_operation(parser);
        VecOperation_push_back(parser->arena, &operations, inner_op);
        parser_expect(parser, TK_NEWLINE);

        // Skip empty lines
        while (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        }
    }
    parser_expect(parser, TK_RBRACE);

    // Pop scope when leaving GPU launch region
    symbol_table_pop_scope(&parser->symbol_table);

    // Create the GPU launch region
    Block *gpu_block = arena_alloc(parser->arena, Block);
    gpu_block->operations = operations.data;
    gpu_block->n_operations = operations.size;
    gpu_block->arguments = launch_args.data;
    gpu_block->n_arguments = launch_args.size;

    Region *gpu_region = arena_alloc(parser->arena, Region);
    gpu_region->blocks = arena_alloc_array(parser->arena, Block*, 1);
    gpu_region->blocks[0] = gpu_block;
    gpu_region->n_blocks = 1;

    op->regions = arena_alloc_array(parser->arena, Region*, 1);
    op->regions[0] = gpu_region;
    op->n_regions = 1;
}

void parse_arith_cmpi(Parser *parser, Operation *op) {
    // Parse arith.cmpi <predicate>, %lhs, %rhs : type
    VecValueRef operands;
    VecValueRef_reserve(parser->arena, &operands, 2);
    
    // Parse predicate (slt, sge, etc.)
    string predicate = str_lit("slt"); // default
    if (parser_peek(parser, TK_NAME)) {
        predicate = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
    }
    
    // Store predicate as an attribute
    Attribute *predicate_attr = arena_alloc(parser->arena, Attribute);
    predicate_attr->name = str_lit("predicate");
    predicate_attr->kind = ATTR_KIND_STRING;
    predicate_attr->data.string_value = predicate;
    
    op->n_attributes = 1;
    op->attributes = arena_alloc_array(parser->arena, Attribute*, 1);
    op->attributes[0] = predicate_attr;
    
    // Expect comma
    parser_expect(parser, TK_COMMA);
    
    // Parse operands
    if (parser_peek(parser, TK_REGISTER)) {
        string lhs_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *lhs = symbol_table_lookup(&parser->symbol_table, lhs_str);
        if (lhs) VecValueRef_push_back(parser->arena, &operands, lhs);
    }
    
    parser_expect(parser, TK_COMMA);
    
    if (parser_peek(parser, TK_REGISTER)) {
        string rhs_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        ValueRef *rhs = symbol_table_lookup(&parser->symbol_table, rhs_str);
        if (rhs) VecValueRef_push_back(parser->arena, &operands, rhs);
    }
    
    // Set operands
    op->operands = operands.data;
    op->n_operands = operands.size;
    // Result type is i1 for arith.cmpi
    op->n_result_types = 1;
    op->result_types = arena_alloc_array(parser->arena, Type*, 1);
    op->result_types[0] = parse_type_from_string(parser->arena, str_lit("i1"));
}
