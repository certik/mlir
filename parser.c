#include <stdbool.h>
#include <stdio.h>

#include "tokenizer.h"
#include <base/arena.h>
#include <base/io.h>
#include "mlir_parser.h"

/*

// MLIR Structures
enum class AttributeKind { ATTR_INT, ATTR_STRING, ATTR_TYPE, ATTR_ARRAY, ATTR_DICT };
using AttrValue = std::variant<long long, std::string, struct Type*, std::vector<struct Attribute*>, std::map<std::string, struct Attribute*>>;
struct Attribute {
    AttributeKind kind;
    AttrValue value;
};

struct Type {
    std::string kind;
    std::map<std::string, Attribute*> attributes;
};

enum class ValueKind { BLOCK_ARG, OP_RESULT };
struct ValueRef {
    ValueKind kind;
    void* def; // Block* or Operation*
    size_t index;
};

struct Operation {
    std::string opname;
    std::vector<ValueRef> operands;
    std::vector<Type*> result_types;
    std::map<std::string, Attribute*> attributes;
    std::vector<struct Region*> regions;
    std::vector<struct Block*> successors;
};

struct Block {
    std::vector<Type*> arg_types;
    std::vector<Operation*> operations;
};

struct Region {
    std::vector<Block*> blocks;
};

*/

// Token
/*
enum class TokenType {
    TOKEN_EOF,
    TOKEN_IDENTIFIER,
    TOKEN_VALUE_ID,
    TOKEN_BLOCK_LABEL,
    TOKEN_INTEGER,
    TOKEN_STRING,
    TOKEN_PUNCTUATION,
    TOKEN_ARROW
};
*/

/*
struct Token {
    TokenType type;
    std::string str_value;
    long long int_value = 0;
};

std::string tokentype_to_string(TokenType tt) {
#define CASE_TOKEN(x) case TokenType::x: return #x;
    switch (tt) {
        CASE_TOKEN(TOKEN_EOF)
        CASE_TOKEN(TOKEN_IDENTIFIER)
        CASE_TOKEN(TOKEN_VALUE_ID)
        CASE_TOKEN(TOKEN_BLOCK_LABEL)
        CASE_TOKEN(TOKEN_INTEGER)
        CASE_TOKEN(TOKEN_STRING)
        CASE_TOKEN(TOKEN_PUNCTUATION)
        CASE_TOKEN(TOKEN_ARROW)
        default: abort();
    }
}


// Lexer
class Lexer {
    const std::string &input;
    size_t pos = 0; // Points to the next (untokenized) character
public:
    Lexer(const std::string& input) : input(input) {}
    Token get_next_token() {
        while (pos < input.size() && std::isspace(input[pos])) pos++;
        if (pos >= input.size()) return {TokenType::TOKEN_EOF, ""};

        char c = input[pos];
        if (std::isalpha(c) || c == '_') {
            size_t start = pos++;
            while (pos < input.size() && (std::isalnum(input[pos]) || input[pos] == '_' || input[pos] == '.')) pos++;
            return {TokenType::TOKEN_IDENTIFIER, input.substr(start, pos - start)};
        }
        if (c == '%') {
            pos++;
            size_t start = pos;
            while (pos < input.size() && std::isalnum(input[pos])) pos++;
            return {TokenType::TOKEN_VALUE_ID, input.substr(start, pos - start)};
        }
        if (c == '^') {
            pos++;
            size_t start = pos;
            while (pos < input.size() && std::isalnum(input[pos])) pos++;
            return {TokenType::TOKEN_BLOCK_LABEL, input.substr(start, pos - start)};
        }
        if (std::isdigit(c)) {
            size_t start = pos;
            while (pos < input.size() && std::isdigit(input[pos])) pos++;
            return {TokenType::TOKEN_INTEGER, "", std::stoll(input.substr(start, pos - start))};
        }
        if (c == '"') {
            pos++;
            size_t start = pos;
            while (pos < input.size() && input[pos] != '"') pos++;
            if (pos >= input.size() || input[pos] != '"') throw std::runtime_error("Unterminated string");
            std::string value = input.substr(start, pos - start);
            pos++;
            return {TokenType::TOKEN_STRING, value};
        }
        if (std::string("{}()[],:=<>").find(c) != std::string::npos) {
            pos++;
            return {TokenType::TOKEN_PUNCTUATION, std::string(1, c)};
        }
        if (c == '-') {
            pos++;
            if (pos >= input.size()) throw std::runtime_error("Unfinished ->");
            c = input[pos];
            if (c == '>') {
                pos++;
                return {TokenType::TOKEN_ARROW, "->"};
            } else {
                throw std::runtime_error("> expected");
            }
        }
        throw std::runtime_error("Unknown character");
    }
};

// Symbol Table
class SymbolTable {
    std::vector<std::unordered_map<std::string, ValueRef>> scopes;
public:
    void push_scope() { scopes.emplace_back(); }
    void pop_scope() { if (!scopes.empty()) scopes.pop_back(); }
    void add_symbol(const std::string& key, ValueRef value) {
        if (scopes.empty()) throw std::runtime_error("No scope");
        scopes.back()[key] = value;
    }
    ValueRef* lookup_symbol(const std::string& key) {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto found = it->find(key);
            if (found != it->end()) return &found->second;
        }
        throw std::runtime_error("Undefined symbol: " + key);
    }
};

// Parser
class Parser {
    Lexer lexer;
    Token current_token;
    SymbolTable symbol_table;

    void consume_token() { current_token = lexer.get_next_token(); }
    void expect_token(TokenType type, const std::string& value = "") {
        if (current_token.type != type || (!value.empty() && current_token.value != value)) {
            throw std::runtime_error("Expected " + (value.empty() ? "token" : value));
        }
        consume_token();
    }

public:
    Parser(const std::string& input) : lexer(input) { current_token = lexer.get_next_token(); }

    Attribute* parse_attribute() {
        if (current_token.type == TokenType::TOKEN_INTEGER) {
            auto attr = new Attribute{AttributeKind::ATTR_INT, current_token.int_value};
            consume_token();
            return attr;
        }
        if (current_token.type == TokenType::TOKEN_STRING) {
            auto attr = new Attribute{AttributeKind::ATTR_STRING, current_token.value};
            consume_token();
            return attr;
        }
        if (current_token.type == TokenType::TOKEN_IDENTIFIER) {
            auto type = parse_type();
            return new Attribute{AttributeKind::ATTR_TYPE, type};
        }
        if (current_token.type == TokenType::TOKEN_PUNCTUATION && current_token.value == "[") {
            consume_token();
            std::vector<Attribute*> elements;
            while (current_token.type != TokenType::TOKEN_PUNCTUATION || current_token.value != "]") {
                elements.push_back(parse_attribute());
                if (current_token.value == ",") consume_token();
            }
            expect_token(TokenType::TOKEN_PUNCTUATION, "]");
            return new Attribute{AttributeKind::ATTR_ARRAY, elements};
        }
        if (current_token.type == TokenType::TOKEN_PUNCTUATION && current_token.value == "{") {
            consume_token();
            std::map<std::string, Attribute*> dict;
            while (current_token.type != TokenType::TOKEN_PUNCTUATION || current_token.value != "}") {
                expect_token(TokenType::TOKEN_IDENTIFIER);
                std::string key = current_token.value;
                consume_token();
                expect_token(TokenType::TOKEN_PUNCTUATION, "=");
                dict[key] = parse_attribute();
                if (current_token.value == ",") consume_token();
            }
            expect_token(TokenType::TOKEN_PUNCTUATION, "}");
            return new Attribute{AttributeKind::ATTR_DICT, dict};
        }
        throw std::runtime_error("Invalid attribute");
    }

    Type* parse_type() {
        expect_token(TokenType::TOKEN_IDENTIFIER);
        auto type = new Type{current_token.value, {}};
        consume_token();
        if (current_token.value == "<") {
            consume_token();
            while (current_token.value != ">") {
                expect_token(TokenType::TOKEN_IDENTIFIER);
                std::string key = current_token.value;
                consume_token();
                expect_token(TokenType::TOKEN_PUNCTUATION, "=");
                type->attributes[key] = parse_attribute();
                if (current_token.value == ",") consume_token();
            }
            expect_token(TokenType::TOKEN_PUNCTUATION, ">");
        }
        return type;
    }

    Operation* parse_operation() {
        auto op = new Operation{{}, {}, {}, {}, {}, {}};
        if (current_token.type == TokenType::TOKEN_VALUE_ID) {
            do {
                std::string name = current_token.value;
                consume_token();
                ValueRef ref{ValueKind::OP_RESULT, op, op->result_types.size()};
                symbol_table.add_symbol(name, ref);
                op->result_types.push_back(nullptr); // Filled later
                if (current_token.value == ",") consume_token();
            } while (current_token.type == TokenType::TOKEN_VALUE_ID);
            expect_token(TokenType::TOKEN_PUNCTUATION, "=");
        }
        expect_token(TokenType::TOKEN_STRING);
        op->opname = current_token.value;
        //consume_token();
        expect_token(TokenType::TOKEN_PUNCTUATION, "(");
        while (current_token.value != ")") {
            expect_token(TokenType::TOKEN_VALUE_ID);
            op->operands.push_back(*symbol_table.lookup_symbol(current_token.value));
            consume_token();
            if (current_token.value == ",") consume_token();
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, ")");
        if (current_token.value == "{") {
            consume_token();
            while (current_token.value != "}") {
                expect_token(TokenType::TOKEN_IDENTIFIER);
                std::string key = current_token.value;
                //consume_token();
                expect_token(TokenType::TOKEN_PUNCTUATION, "=");
                op->attributes[key] = parse_attribute();
                if (current_token.value == ",") consume_token();
            }
            expect_token(TokenType::TOKEN_PUNCTUATION, "}");
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, ":");
        expect_token(TokenType::TOKEN_PUNCTUATION, "(");
        while (current_token.value != ")") {
            op->result_types.push_back(parse_type());
            if (current_token.value == ",") consume_token();
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, ")");
        expect_token(TokenType::TOKEN_PUNCTUATION, "->");
        op->result_types[0] = parse_type();
        return op;
    }

    Block* parse_block() {
        auto block = new Block{{}, {}};
        if (current_token.type == TokenType::TOKEN_BLOCK_LABEL) {
            consume_token();
            expect_token(TokenType::TOKEN_PUNCTUATION, ":");
        }
        while (current_token.type != TokenType::TOKEN_EOF && current_token.value != "}") {
            block->operations.push_back(parse_operation());
        }
        return block;
    }

    Region* parse_region() {
        expect_token(TokenType::TOKEN_PUNCTUATION, "{");
        symbol_table.push_scope();
        std::vector<Block*> blocks;
        while (current_token.value != "}") {
            blocks.push_back(parse_block());
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, "}");
        symbol_table.pop_scope();
        return new Region{blocks};
    }

    Operation* parse_module() {
        expect_token(TokenType::TOKEN_IDENTIFIER, "module");
        expect_token(TokenType::TOKEN_PUNCTUATION, "{");
        symbol_table.push_scope();
        std::vector<Operation*> ops;
        while (current_token.value != "}") {
            ops.push_back(parse_operation());
        }
        expect_token(TokenType::TOKEN_PUNCTUATION, "}");
        symbol_table.pop_scope();
        auto block = new Block{{}, ops};
        auto region = new Region{{block}};
        return new Operation{"builtin.module", {}, {}, {}, {region}, {}};
    }
};
*/

void tokenizer_print_all_tokens(Arena *arena, const string input_code) {
    unsigned char *string_start;
    string_start = (unsigned char*)input_code.str;
    uint64_t cur=0;
    while (true) {
        TokenType token_type;
        uint64_t first, last;
        first = cur;
        tokenizer_get_next_token(string_start, &cur, &token_type);
        last = cur-1;
        bool debug = false;
        if (debug) {
            string token_str = str_substr(input_code, first, last-first+1);
            if (token_type == TK_WHITESPACE || token_type == TK_NEWLINE
                    || token_type == TK_EOF) {
                token_str = str_lit("");
            }
            println(arena, str_lit("Token({}, \"{}\", {},{})"),
                tokentype_to_string(token_type), token_str, first, last);
        }
        if (token_type == TK_EOF) {
            return;
        }
    }
}

string print_operation(Arena *arena, int indent_level, Operation *op);
static const char* op_type_to_string(OpType type);

string indent(Arena *arena, int indent_level) {
    const int indent_spaces=4;
    int buf_size=indent_level*indent_spaces;
    char* buf = arena_alloc_array(arena, char, buf_size);
    for (int64_t i = 0; i < buf_size; i++) {
        buf[i] = ' ';
    }
    string str = {buf, buf_size};
    return str;
}

string print_block(Arena *arena, int bb_index, int indent_level, Block *block) {
    string result = format(arena, str_lit("{}^bb{}"), indent(arena, indent_level), bb_index);

    // Print block arguments if any
    if (block->n_arguments > 0 && block->arguments) {
        result = str_concat(arena, result, str_lit("("));
        for (int i = 0; i < block->n_arguments; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            ValueRef *arg = block->arguments[i];
            if (arg && arg->type) {
                // For block arguments, use the original register name
                if (arg->register_name.size > 0) {
                    result = str_concat(arena, result, format(arena, str_lit("{}: {}"),
                                                            arg->register_name, arg->type->str));
                } else {
                    result = str_concat(arena, result, format(arena, str_lit("%arg{}: {}"),
                                                            (int64_t)arg->result_index, arg->type->str));
                }
            } else {
                result = str_concat(arena, result, str_lit("null_arg"));
            }
        }
        result = str_concat(arena, result, str_lit(")"));
    }

    result = str_concat(arena, result, str_lit(":\n"));

    for (int i=0; i < block->n_operations; i++) {
        result = str_concat(arena, result,
            print_operation(arena, indent_level+1, block->operations[i])
            );
    }
    return result;
}

string print_region(Arena *arena, int indent_level, Region *region) {
    string result = str_lit("");
    result = str_concat(arena, result, str_lit("{\n"));
    for (int i=0; i < region->n_blocks; i++) {
        result = str_concat(arena, result,
            print_block(arena, i, indent_level, region->blocks[i])
            );
    }
    result = str_concat(arena, result, indent(arena, indent_level));
    result = str_concat(arena, result, str_lit("}"));
    return result;
}

int reg_idx=0;

static const char* op_type_to_string(OpType type) {
    switch (type) {
        case OP_TYPE_UNREGISTERED: return "unregistered";
        case OP_TYPE_MODULE: return "module";
        case OP_TYPE_ARITH_ADDI: return "arith.addi";
        case OP_TYPE_ARITH_SUBI: return "arith.subi";
        case OP_TYPE_ARITH_MULI: return "arith.muli";
        case OP_TYPE_ARITH_DIVI: return "arith.divi";
        case OP_TYPE_ARITH_ADDF: return "arith.addf";
        case OP_TYPE_ARITH_SUBF: return "arith.subf";
        case OP_TYPE_ARITH_MULF: return "arith.mulf";
        case OP_TYPE_ARITH_DIVF: return "arith.divf";
        case OP_TYPE_ARITH_CONSTANT: return "arith.constant";
        case OP_TYPE_ARITH_CMPI: return "arith.cmpi";
        case OP_TYPE_ARITH_CMPF: return "arith.cmpf";
        case OP_TYPE_MEMREF_LOAD: return "memref.load";
        case OP_TYPE_MEMREF_STORE: return "memref.store";
        case OP_TYPE_MEMREF_ALLOC: return "memref.alloc";
        case OP_TYPE_MEMREF_DEALLOC: return "memref.dealloc";
        case OP_TYPE_CF_BR: return "cf.br";
        case OP_TYPE_CF_COND_BR: return "cf.cond_br";
        case OP_TYPE_CF_SWITCH: return "cf.switch";
        case OP_TYPE_FUNC_FUNC: return "func.func";
        case OP_TYPE_FUNC_RETURN: return "func.return";
        case OP_TYPE_FUNC_CALL: return "func.call";
        case OP_TYPE_SCF_FOR: return "scf.for";
        case OP_TYPE_SCF_WHILE: return "scf.while";
        case OP_TYPE_SCF_IF: return "scf.if";
        case OP_TYPE_TT_GET_PROGRAM_ID: return "tt.get_program_id";
        default: return "unknown";
    }
}

string print_operation(Arena *arena, int indent_level, Operation *op) {
    string result = indent(arena, indent_level);

    // Print results if any
    if (op->n_result_types > 0) {
        for (int i = 0; i < op->n_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));

            // Use SSA number from operation's results if available
            if (op->n_results > i && op->results && op->results[i]) {
                result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)op->results[i]->ssa_number));
            } else {
                // Fallback to global counter
                result = str_concat(arena, result, format(arena, str_lit("%{}"), reg_idx + i));
            }
        }
        result = str_concat(arena, result, str_lit(" = "));
        if (!op->n_results || !op->results) {
            reg_idx += op->n_result_types;
        }
    }

    // Print operation name (quotes only for unregistered operations, except tt.func)
    bool is_tt_func = (op->opname.size > 0 && str_eq(op->opname, str_lit("tt.func")));
    if (op->op_type == OP_TYPE_UNREGISTERED && !is_tt_func) {
        result = str_concat(arena, result, str_lit("\""));
        if (op->opname.size > 0) {
            result = str_concat(arena, result, op->opname);
        } else {
            result = str_concat(arena, result, str_lit("unknown"));
        }
        result = str_concat(arena, result, str_lit("\""));
    } else {
        if (op->opname.size > 0) {
            result = str_concat(arena, result, op->opname);
        } else {
            result = str_concat(arena, result, str_from_cstr_view((char*)op_type_to_string(op->op_type)));
        }
    }

    // Print operands with types (always include parentheses)
    result = str_concat(arena, result, str_lit("("));
    for (int i = 0; i < op->n_operands; i++) {
        if (i > 0) result = str_concat(arena, result, str_lit(", "));
        ValueRef *operand = op->operands[i];
        if (operand == NULL) {
            result = str_concat(arena, result, str_lit("NULL_OPERAND"));
            continue;
        }
        // Use original register name for BLOCK_ARG, SSA number for OP_RESULT
        if (operand->kind == BLOCK_ARG && operand->register_name.size > 0) {
            result = str_concat(arena, result, operand->register_name);
        } else if (operand->ssa_number < 1000) {
            result = str_concat(arena, result, format(arena, str_lit("%{}"), (int64_t)operand->ssa_number));
        } else {
            // Fallback to register name for debugging unresolved values
            result = str_concat(arena, result, operand->register_name);
        }
        result = str_concat(arena, result, str_lit(": "));
        result = str_concat(arena, result, operand->type->str);
    }
    result = str_concat(arena, result, str_lit(")"));

    // Print attributes if any (skip for tt.get_program_id as it's handled specially)
    if (op->n_attributes > 0) {
        result = str_concat(arena, result, str_lit(" {"));
        for (int i = 0; i < op->n_attributes; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            Attribute *attr = op->attributes[i];
            result = str_concat(arena, result, format(arena, str_lit("{} = "), attr->name));
            switch (attr->kind) {
                case ATTR_KIND_INTEGER:
                    // Add type annotation for tt.make_range attributes
                    if (str_eq(op->opname, str_lit("tt.make_range"))) {
                        result = str_concat(arena, result, format(arena, str_lit("{} : i32"), attr->data.integer_value));
                    } else {
                        result = str_concat(arena, result, format(arena, str_lit("{}"), attr->data.integer_value));
                    }
                    break;
                case ATTR_KIND_STRING:
                    result = str_concat(arena, result, format(arena, str_lit("\"{}\""), attr->data.string_value));
                    break;
                default:
                    result = str_concat(arena, result, str_lit("..."));
            }
        }
        result = str_concat(arena, result, str_lit("}"));
    }

    // Print result types if any
    if (op->n_result_types > 0) {
        result = str_concat(arena, result, str_lit(" -> "));
        for (int i = 0; i < op->n_result_types; i++) {
            if (i > 0) result = str_concat(arena, result, str_lit(", "));
            if (op->result_types && op->result_types[i]) {
                result = str_concat(arena, result, op->result_types[i]->str);
            } else {
                result = str_concat(arena, result, str_lit("?"));
            }
        }
    }

    // Print regions if any
    if (op->n_regions > 0) {
        result = str_concat(arena, result, str_lit(" "));
        for (int i = 0; i < op->n_regions; i++) {
            result = str_concat(arena, result,
                print_region(arena, indent_level, op->regions[i])
                );
        }
    }

    result = str_concat(arena, result, str_lit("\n"));
    return result;
}

// Main
Operation* construct_test_module(Arena *arena) {
    // Create simple module operation
    Operation *module = arena_alloc(arena, Operation);
    module->op_type = OP_TYPE_MODULE;
    module->operands = NULL;
    module->n_operands = 0;
    module->result_types = NULL;
    module->n_result_types = 0;
    module->attributes = NULL;
    module->n_attributes = 0;
    module->results = NULL;
    module->n_results = 0;
    module->opname = str_lit("module");

    // Create module region and block (empty)
    Region *module_region = arena_alloc(arena, Region);
    Block *module_block = arena_alloc(arena, Block);
    module_block->arguments = NULL;
    module_block->n_arguments = 0;
    module_block->operations = NULL;
    module_block->n_operations = 0;

    module_region->n_blocks = 1;
    module_region->blocks = arena_alloc_array(arena, Block*, 1);
    module_region->blocks[0] = module_block;

    module->n_regions = 1;
    module->regions = arena_alloc_array(arena, Region*, 1);
    module->regions[0] = module_region;

    return module;
}

Operation* construct_test_module_full(Arena *arena) {
    // Create types
    Type *i32_type = arena_alloc(arena, Type);
    i32_type->kind = TYPE_KIND_INTEGER;
    i32_type->data.integer.width = 32;
    i32_type->data.integer.is_signed = true;
    i32_type->str = str_lit("i32");

    Type *i64_type = arena_alloc(arena, Type);
    i64_type->kind = TYPE_KIND_INTEGER;
    i64_type->data.integer.width = 64;
    i64_type->data.integer.is_signed = true;
    i64_type->str = str_lit("i64");

    // Create module operation
    Operation *module = arena_alloc(arena, Operation);
    module->op_type = OP_TYPE_MODULE;
    module->operands = NULL;
    module->n_operands = 0;
    module->result_types = NULL;
    module->n_result_types = 0;
    module->attributes = NULL;
    module->n_attributes = 0;
    module->results = NULL;
    module->n_results = 0;
    module->opname = str_lit("module");

    // Create module region
    Region *module_region = arena_alloc(arena, Region);
    Block *module_block = arena_alloc(arena, Block);
    module_block->arguments = NULL;
    module_block->n_arguments = 0;

    // Create function operation
    Operation *func_op = arena_alloc(arena, Operation);
    func_op->op_type = OP_TYPE_FUNC_FUNC;
    func_op->operands = NULL;
    func_op->n_operands = 0;
    func_op->result_types = NULL;
    func_op->n_result_types = 0;
    func_op->results = NULL;
    func_op->n_results = 0;
    func_op->opname = str_lit("func.func");

    // Function attributes (sym_name)
    func_op->n_attributes = 1;
    func_op->attributes = arena_alloc_array(arena, Attribute*, 1);
    func_op->attributes[0] = arena_alloc(arena, Attribute);
    func_op->attributes[0]->kind = ATTR_KIND_STRING;
    func_op->attributes[0]->data.string_value = str_lit("example_func");
    func_op->attributes[0]->name = str_lit("sym_name");

    // Create function region and block
    Region *func_region = arena_alloc(arena, Region);
    Block *func_block = arena_alloc(arena, Block);

    // Function block arguments (%arg0, %arg1)
    func_block->n_arguments = 2;
    func_block->arguments = arena_alloc_array(arena, ValueRef*, 2);
    func_block->arguments[0] = create_value_ref(arena, BLOCK_ARG);
    func_block->arguments[0]->result_index = 0;
    func_block->arguments[0]->type = i32_type;
    func_block->arguments[0]->register_name = str_lit("%arg0");
    func_block->arguments[1] = create_value_ref(arena, BLOCK_ARG);
    func_block->arguments[1]->result_index = 1;
    func_block->arguments[1]->type = i32_type;
    func_block->arguments[1]->register_name = str_lit("%arg1");

    // Create operations in function block
    func_block->n_operations = 4;
    func_block->operations = arena_alloc_array(arena, Operation*, 4);

    // %0 = arith.constant 5 : i32
    Operation *const_op = arena_alloc(arena, Operation);
    const_op->op_type = OP_TYPE_ARITH_CONSTANT;
    const_op->operands = NULL;
    const_op->n_operands = 0;
    const_op->n_result_types = 1;
    const_op->result_types = arena_alloc_array(arena, Type*, 1);
    const_op->result_types[0] = i32_type;
    const_op->n_attributes = 1;
    const_op->attributes = arena_alloc_array(arena, Attribute*, 1);
    const_op->attributes[0] = arena_alloc(arena, Attribute);
    const_op->attributes[0]->kind = ATTR_KIND_INTEGER;
    const_op->attributes[0]->data.integer_value = 5;
    const_op->attributes[0]->name = str_lit("value");
    const_op->regions = NULL;
    const_op->n_regions = 0;

    // Create const_result before linking
    ValueRef *const_result = create_value_ref(arena, OP_RESULT);
    const_result->def = const_op;
    const_result->result_index = 0;
    const_result->type = i32_type;
    const_result->register_name = str_lit("%0");
    const_result->ssa_number = 0;

    const_op->results = arena_alloc_array(arena, ValueRef*, 1);
    const_op->results[0] = const_result;
    const_op->n_results = 1;
    const_op->opname = str_lit("arith.constant");

    // %1 = arith.addi %arg0, %arg1 : i32
    Operation *add_op = arena_alloc(arena, Operation);
    add_op->op_type = OP_TYPE_ARITH_ADDI;
    add_op->n_operands = 2;
    add_op->operands = arena_alloc_array(arena, ValueRef*, 2);
    add_op->operands[0] = func_block->arguments[0];
    add_op->operands[1] = func_block->arguments[1];
    add_op->n_result_types = 1;
    add_op->result_types = arena_alloc_array(arena, Type*, 1);
    add_op->result_types[0] = i32_type;
    add_op->attributes = NULL;
    add_op->n_attributes = 0;
    add_op->regions = NULL;
    add_op->n_regions = 0;
    // Create add_result before linking
    ValueRef *add_result = create_value_ref(arena, OP_RESULT);
    add_result->def = add_op;
    add_result->result_index = 1;
    add_result->type = i32_type;
    add_result->register_name = str_lit("%1");
    add_result->ssa_number = 1;

    add_op->results = arena_alloc_array(arena, ValueRef*, 1);
    add_op->results[0] = add_result;
    add_op->n_results = 1;
    add_op->opname = str_lit("arith.addi");

    // %2 = arith.muli %1, %0 : i32 (add_result and const_result already created above)

    // Create mul_result before creating mul_op
    ValueRef *mul_result = create_value_ref(arena, OP_RESULT);
    mul_result->result_index = 2;  // This will be %2
    mul_result->type = i32_type;
    mul_result->register_name = str_lit("%2");
    mul_result->ssa_number = 2;

    Operation *mul_op = arena_alloc(arena, Operation);
    mul_op->op_type = OP_TYPE_ARITH_MULI;
    mul_op->n_operands = 2;
    mul_op->operands = arena_alloc_array(arena, ValueRef*, 2);
    mul_op->operands[0] = add_result;
    mul_op->operands[1] = const_result;
    mul_op->n_result_types = 1;
    mul_op->result_types = arena_alloc_array(arena, Type*, 1);
    mul_op->result_types[0] = i32_type;
    mul_op->attributes = NULL;
    mul_op->n_attributes = 0;
    mul_op->regions = NULL;
    mul_op->n_regions = 0;
    mul_op->results = arena_alloc_array(arena, ValueRef*, 1);
    mul_op->results[0] = mul_result;
    mul_op->n_results = 1;
    mul_op->opname = str_lit("arith.muli");

    // Set def pointer for mul_result now that mul_op exists
    mul_result->def = mul_op;

    // func.return %2 : i32

    Operation *ret_op = arena_alloc(arena, Operation);
    ret_op->op_type = OP_TYPE_FUNC_RETURN;
    ret_op->n_operands = 1;
    ret_op->operands = arena_alloc_array(arena, ValueRef*, 1);
    ret_op->operands[0] = mul_result;
    ret_op->result_types = NULL;
    ret_op->n_result_types = 0;
    ret_op->attributes = NULL;
    ret_op->n_attributes = 0;
    ret_op->regions = NULL;
    ret_op->n_regions = 0;
    ret_op->results = NULL;
    ret_op->n_results = 0;
    ret_op->opname = str_lit("func.return");

    // Link operations to function block
    func_block->operations[0] = const_op;
    func_block->operations[1] = add_op;
    func_block->operations[2] = mul_op;
    func_block->operations[3] = ret_op;

    // Link function block to function region
    func_region->n_blocks = 1;
    func_region->blocks = arena_alloc_array(arena, Block*, 1);
    func_region->blocks[0] = func_block;

    // Link function region to function operation
    func_op->n_regions = 1;
    func_op->regions = arena_alloc_array(arena, Region*, 1);
    func_op->regions[0] = func_region;

    // Link function operation to module block
    module_block->n_operations = 1;
    module_block->operations = arena_alloc_array(arena, Operation*, 1);
    module_block->operations[0] = func_op;

    // Link module block to module region
    module_region->n_blocks = 1;
    module_region->blocks = arena_alloc_array(arena, Block*, 1);
    module_region->blocks[0] = module_block;

    // Link module region to module operation
    module->n_regions = 1;
    module->regions = arena_alloc_array(arena, Region*, 1);
    module->regions[0] = module_region;

    return module;
}

int main(int argc, char *argv[]) {
    printf("Starting main...\n");
    Arena *arena = arena_create(50*1024*1024);  // Increase arena size
    printf("Arena created...\n");

    // Check for --construct option
    bool use_construction = false;
    char *input_file = NULL;

    printf("Parsing args...\n");
    for (int i = 1; i < argc; i++) {
        printf("Arg %d: %s\n", i, argv[i]);
        if (strcmp(argv[i], "--construct") == 0) {
            use_construction = true;
            printf("Construction mode enabled\n");
        } else if (argv[i][0] != '-') {
            input_file = argv[i];
        }
    }
    printf("Done parsing args. use_construction=%d\n", use_construction);

    Operation* op;

    int exit_code;
    if (use_construction) {
        // Use constructed test module
        printf("Creating module...\n");
        op = construct_test_module_full(arena);
        printf("Module created successfully.\n");

        // Test generic printing with expected output comparison
        printf("=== Generic Printer Test ===\n");
        reg_idx = 0;  // Reset SSA counter
        printf("About to print operation...\n");
        string result = print_operation(arena, 0, op);
        printf("Printing result...\n");
        println(arena, str_lit("{}"), result);

        // Reference expected output for generic mode
        const char *expected =
            "module() {\n"
            "^bb0:\n"
            "    func.func() {sym_name = \"example_func\"} {\n"
            "    ^bb0(%arg0: i32, %arg1: i32):\n"
            "        %0 = arith.constant() {value = 5} -> i32\n"
            "        %1 = arith.addi(%arg0: i32, %arg1: i32) -> i32\n"
            "        %2 = arith.muli(%1: i32, %0: i32) -> i32\n"
            "        func.return(%2: i32)\n"
            "    }\n"
            "}\n";

        // Compare output
        if (str_eq(result, str_from_cstr_view((char*)expected))) {
            printf("✅ Generic mode test PASSED\n");
            exit_code = 0;
        } else {
            printf("❌ Generic mode test FAILED\n");
            printf("Expected:\n%s\n", expected);
            printf("Actual:\n");
            println(arena, str_lit("{}"), result);
            exit_code = 1;
        }
    } else {
        // Use parser mode
        string mlir_code = str_lit("module {\n"
                                "  %0 = \"std.constant\"() {value = 42} : () -> i32\n"
                                "  \"std.return\"(%0) : (i32) -> ()\n"
                                "}");

        if (input_file) {
            mlir_code = read_file_ok(arena, str_from_cstr_view(input_file));
        }

        tokenizer_print_all_tokens(arena, mlir_code);

        Parser parser;
        parser_init(arena, &parser, mlir_code);
        op = parse_module(&parser);
        println(arena, str_lit("MLIR:"));
        reg_idx = 0;  // Reset SSA counter
        println(arena, str_lit("{}"), print_operation(arena, 0, op));
        exit_code = 0;
    }

    arena_free(arena);
    return exit_code;
}
