"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UniswapLP = void 0;
const error_handler_1 = require("../../services/error-handler");
const uniswap_config_1 = require("./uniswap.config");
const ethereum_1 = require("../../chains/ethereum/ethereum");
const uniswap_lp_helper_1 = require("./uniswap.lp.helper");
const config_manager_v2_1 = require("../../services/config-manager-v2");
const ethers_1 = require("ethers");
class UniswapLP {
    constructor(chain, network) {
        this._wallet = null;
        this._burnToken = false;
        this._burnTokenPositionIndex = 0;
        this._lpHelper = new uniswap_lp_helper_1.UniswapLPHelper(chain, network);
        this._chain = chain;
        this._network = network;
        this._token0 = null;
        this._token1 = null;
        this._amount0Desired = '0';
        this._amount1Desired = '0';
        this._fee = config_manager_v2_1.defaultFee(this._network, this._chain);
        this._lowerPrice = '0';
        this._upperPrice = '0';
        this._decreasePercent = '0';
        this._tokenId = 0;
        this._deadline = this._lpHelper.ttl;
        this._wallet = null;
    }
    init() {
        return this._lpHelper.init();
    }
    setToken0(token0) {
        this._token0 = token0;
        return this;
    }
    setToken1(token1) {
        this._token1 = token1;
        return this;
    }
    setAmount0Desired(amount0) {
        this._amount0Desired = amount0;
        return this;
    }
    setAmount1Desired(amount1) {
        this._amount1Desired = amount1;
        return this;
    }
    setFee(fee) {
        this._fee = fee;
        return this;
    }
    setLowerPrice(lowerPrice) {
        this._lowerPrice = lowerPrice;
        return this;
    }
    setUpperPrice(upperPrice) {
        this._upperPrice = upperPrice;
        return this;
    }
    setDeadline(deadline) {
        this._deadline = deadline;
        return this;
    }
    setWallet(wallet) {
        this._wallet = wallet;
        return this;
    }
    setBurnToken(burnToken, positionIndex = 0) {
        this._burnToken = burnToken;
        this._burnTokenPositionIndex = positionIndex;
        return this;
    }
    setDecreasePercent(percent) {
        this._decreasePercent = percent;
        return this;
    }
    setTokenId(tokenId) {
        this._tokenId = tokenId;
        return this;
    }
    static log(message) {
        console.log(`[UniswapLP] ${message}`);
    }
    static error(message) {
        console.error(`[UniswapLP] ERROR: ${message}`);
    }
    validate() {
        if (!this._lpHelper.ready()) {
            throw new error_handler_1.InitializationError((0, error_handler_1.SERVICE_UNITIALIZED_ERROR_MESSAGE)('UniswapLPHelper'), error_handler_1.SERVICE_UNITIALIZED_ERROR_CODE);
        }
        if (this._fee < 500 || this._fee > 3000) {
            throw new Error('Invalid fee. Please provide fee between 500 and 3000.');
        }
        if (this._wallet === null) {
            throw new Error('Wallet not set. Please use setWallet method to set the wallet.');
        }
        if (!this._wallet.provider) {
            throw new Error('Invalid wallet provider.');
        }
    }
    addLiquidity() {
        var _a;
        return __awaiter(this, void 0, void 0, function* () {
            try {
                this.validate();
                if (this._token0 === null || this._token1 === null) {
                    throw new Error('Token0 and Token1 must be set.');
                }
                const walletAddress = this._wallet.address;
                UniswapLP.log(`Adding liquidity to Uniswap for ${this._token0.symbol} and ${this._token1.symbol}...`);
                const amount0Min = '0';
                const amount1Min = '0';
                const { addCallParameters, swapRequired } = yield this._lpHelper.addPositionHelper(this._wallet, this._token0, this._token1, this._amount0Desired, this._amount1Desired, this._fee, this._lowerPrice, this._upperPrice, this._tokenId);
                if (swapRequired) {
                    UniswapLP.log(`Swapping tokens before adding liquidity...`);
                    const transaction = this._lpHelper.alphaRouter.swapCallParameters(addCallParameters);
                    yield ((_a = this._wallet) === null || _a === void 0 ? void 0 : _a.sendTransaction(transaction));
                }
                UniswapLP.log(`Adding liquidity to Uniswap...`);
                const contract = this._lpHelper.getContract('nft', this._wallet);
                const { events } = yield contract.addLiquidity(addCallParameters, {
                    gasLimit: uniswap_config_1.UniswapConfig.config.gasLimit,
                    value: uniswap_config_1.UniswapConfig.config.value,
                });
                UniswapLP.log(`Liquidity added successfully!`);
                const nftTransferEvent = events.find((event) => event.event === 'Transfer');
                if (nftTransferEvent) {
                    const tokenId = nftTransferEvent.args.tokenId.toNumber();
                    UniswapLP.log(`New NFT position created with tokenId: ${tokenId}`);
                    return tokenId;
                }
                else {
                    UniswapLP.error('No NFT Transfer event found after adding liquidity.');
                    throw new Error('Failed to add liquidity.');
                }
            }
            catch (error) {
                UniswapLP.error(error.message);
                throw error;
            }
        });
    }
    reduceLiquidity() {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                this.validate();
                if (this._tokenId === 0) {
                    throw new Error('tokenId must be set for reducing liquidity.');
                }
                UniswapLP.log(`Reducing liquidity from Uniswap for tokenId: ${this._tokenId}...`);
                const { removeCallParameters } = yield this._lpHelper.reducePositionHelper(this._wallet, this._tokenId, this._decreasePercent);
                UniswapLP.log(`Removing liquidity from Uniswap...`);
                const contract = this._lpHelper.getContract('nft', this._wallet);
                yield contract.decreaseLiquidity(removeCallParameters, {
                    gasLimit: uniswap_config_1.UniswapConfig.config.gasLimit,
                    value: uniswap_config_1.UniswapConfig.config.value,
                });
                UniswapLP.log(`Liquidity removed successfully!`);
            }
            catch (error) {
                UniswapLP.error(error.message);
                throw error;
            }
        });
    }
}
exports.UniswapLP = UniswapLP;
//# sourceMappingURL=uniswap.lp.js.map